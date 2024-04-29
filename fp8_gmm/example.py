import math
from time import time

import megablocks.ops as mega_ops
import numpy as np
import torch
import transformer_engine.pytorch as tex
from megablocks import grouped_gemm_util as gg_util
from transformer_engine.common import recipe

from fp8_gmm.grouped_linear import GroupedLinear


def topk(scores, top_k, moe_renorm=False):
    assert top_k in [1, 2], "only top-1/2 gating has been tested!"

    weights = torch.nn.functional.softmax(
        scores,
        dim=1,
        dtype=torch.float,
    )
    multiplier, selected_experts = torch.topk(weights, top_k)
    if moe_renorm:
        multiplier = multiplier / multiplier.sum(dim=-1, keepdim=True)

    return multiplier, selected_experts


def indices_and_bins(top_expert, sort_end_bit, n_experts):
    top_expert = top_expert.int()
    bin_ids, indices = mega_ops.sort(top_expert, sort_end_bit)
    tokens_per_expert = mega_ops.histogram(top_expert, n_experts)
    bins = mega_ops.inclusive_cumsum(tokens_per_expert, 0)
    bins = bins.view(1) if not len(bins.size()) else bins
    return indices, bin_ids, bins, tokens_per_expert


class Bf16Module(torch.nn.Module):
    def __init__(self, n_embd, n_inner, n_experts, top_k):
        super(Bf16Module, self).__init__()
        self.n_embd = n_embd
        self.n_inner = n_inner
        self.n_experts = n_experts
        self.top_k = top_k
        self.sort_end_bit = max(int(np.ceil(np.log2(n_experts))), 1)
        self.gating_network = torch.nn.Linear(n_embd, n_experts, bias=False)
        self.fc1 = torch.nn.Linear(n_embd, n_inner * n_experts, bias=False)
        self.fc2 = torch.nn.Linear(n_embd, n_inner * n_experts, bias=False)
        torch.nn.init.uniform_(self.fc1.weight.data, -math.sqrt(1.0 / n_embd), math.sqrt(1.0 / n_embd))
        torch.nn.init.uniform_(self.fc2.weight.data, -math.sqrt(1.0 / n_inner), math.sqrt(1.0 / n_inner))
        self.act = torch.nn.GELU()

    def compute(self, x, tokens_per_expert, indices, bin_ids, expert_weights, bins, top_k):
        x = x.view(-1, x.size(-1))
        x = mega_ops.gather(x, indices, bin_ids, bins, top_k)
        batch_sizes = tokens_per_expert.cpu().to(torch.long)
        w1 = self.fc1.weight.view(self.n_experts, -1, self.n_embd)
        w2 = self.fc2.weight.view(self.n_experts, -1, self.n_embd)
        x = gg_util.ops.gmm(x, w1, batch_sizes, trans_b=True)
        x = self.act(x)
        out = gg_util.ops.gmm(x, w2, batch_sizes)
        return mega_ops.scatter(out, indices, bin_ids, expert_weights, bins, top_k)

    def forward(self, input):
        input = input.view(-1, input.size(-1))
        routing_logits = self.gating_network(input)
        weights0, selected_experts = topk(routing_logits, self.top_k)
        expert_weights = weights0.flatten()
        top_experts = selected_experts.flatten()
        with torch.no_grad():
            indices, bin_ids, bins, token_per_expert = indices_and_bins(top_experts, self.sort_end_bit, self.n_experts)
        return self.compute(input, token_per_expert, indices, bin_ids, expert_weights, bins, self.top_k)


class Fp8Module(torch.nn.Module):
    def __init__(self, n_embd, n_inner, n_experts, top_k, dtype=torch.bfloat16, cutlass=False):
        super(Fp8Module, self).__init__()
        self.n_embd = n_embd
        self.n_inner = n_inner
        self.n_experts = n_experts
        self.top_k = top_k
        self.sort_end_bit = max(int(np.ceil(np.log2(n_experts))), 1)
        self.gating_network = torch.nn.Linear(n_embd, n_experts, bias=False).to(dtype).to("cuda")
        self.grouped_linear1 = GroupedLinear(n_embd, n_inner, n_experts, dtype=dtype, cutlass=cutlass)
        self.grouped_linear2 = GroupedLinear(n_inner, n_embd, n_experts, dtype=dtype, cutlass=cutlass)
        self.act = torch.nn.GELU()

    def compute(self, x, tokens_per_expert, indices, bin_ids, expert_weights, bins, top_k):
        x = x.view(-1, x.size(-1))
        x = mega_ops.gather(x, indices, bin_ids, bins, top_k)
        batch_sizes = tokens_per_expert.cpu().to(torch.long)
        x = self.grouped_linear1(x, batch_sizes)
        x = self.act(x)
        out = self.grouped_linear2(x, batch_sizes)
        return mega_ops.scatter(out, indices, bin_ids, expert_weights, bins, top_k)

    def forward(self, input):
        input = input.view(-1, input.size(-1))
        routing_logits = self.gating_network(input)
        weights0, selected_experts = topk(routing_logits, self.top_k)
        expert_weights = weights0.flatten()
        top_experts = selected_experts.flatten()
        with torch.no_grad():
            indices, bin_ids, bins, token_per_expert = indices_and_bins(top_experts, self.sort_end_bit, self.n_experts)
        return self.compute(input, token_per_expert, indices, bin_ids, expert_weights, bins, self.top_k)


num_tokens, hidden_states, num_inner, num_experts, top_k = 16384, 4096, 6400, 16, 2
dtype = torch.bfloat16
warmup_steps, run_steps = 4, 16

# Bf16Module run.
model_bf16 = Bf16Module(hidden_states, num_inner, num_experts, top_k).to(dtype).to("cuda")
for _ in range(warmup_steps):
    input = torch.randn(num_tokens, hidden_states, dtype=dtype, device="cuda", requires_grad=True)
    out = model_bf16(input)
    out.backward(torch.randn(*out.size(), dtype=dtype, device="cuda"))
torch.cuda.synchronize()
start = time()
for _ in range(run_steps):
    input = torch.randn(num_tokens, hidden_states, dtype=dtype, device="cuda", requires_grad=True)
    out = model_bf16(input)
    out.backward(torch.randn(*out.size(), dtype=dtype, device="cuda"))
torch.cuda.synchronize()
print(f"Bf16Module run: {(time() - start) * 1000 / run_steps:.3f} ms")

# Fp8Module, using cublasLtMatmul.
model_fp8_cublas = Fp8Module(hidden_states, num_inner, num_experts, top_k, dtype=dtype, cutlass=False)
fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.HYBRID)
for _ in range(warmup_steps):
    input = torch.randn(num_tokens, hidden_states, dtype=dtype, device="cuda", requires_grad=True)
    with tex.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        out = model_fp8_cublas(input)
    out.backward(torch.randn(*out.size(), dtype=dtype, device="cuda"))
torch.cuda.synchronize()
start = time()
for _ in range(run_steps):
    input = torch.randn(num_tokens, hidden_states, dtype=dtype, device="cuda", requires_grad=True)
    with tex.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        out = model_fp8_cublas(input)
    out.backward(torch.randn(*out.size(), dtype=dtype, device="cuda"))
torch.cuda.synchronize()
print(f"Fp8Module cublasLtMatmul run: {(time() - start) * 1000 / run_steps:.3f} ms")

# Fp8Module, using cutlass grouped gemm.
model_fp8_cutlass = Fp8Module(hidden_states, num_inner, num_experts, top_k, dtype=dtype, cutlass=True)
for _ in range(warmup_steps):
    input = torch.randn(num_tokens, hidden_states, dtype=dtype, device="cuda", requires_grad=True)
    with tex.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        out = model_fp8_cutlass(input)
    out.backward(torch.randn(*out.size(), dtype=dtype, device="cuda"))
torch.cuda.synchronize()
start = time()
for _ in range(run_steps):
    input = torch.randn(num_tokens, hidden_states, dtype=dtype, device="cuda", requires_grad=True)
    with tex.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
        out = model_fp8_cutlass(input)
    out.backward(torch.randn(*out.size(), dtype=dtype, device="cuda"))
torch.cuda.synchronize()
print(f"Fp8Module cutlass run: {(time() - start) * 1000 / run_steps:.3f} ms")
