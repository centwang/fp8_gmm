import math

import megablocks.ops as mega_ops
import numpy as np
import torch
import transformer_engine.pytorch as tex
from megablocks import grouped_gemm_util as gg_util
from transformer_engine.common import recipe

from fp8_gmm import ops as fp8_gmm_ops


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
    def __init__(self, n_embd, n_inner, n_experts, top_k):
        super(Fp8Module, self).__init__()
        self.n_embd = n_embd
        self.n_inner = n_inner
        self.n_experts = n_experts
        self.top_k = top_k
        self.sort_end_bit = max(int(np.ceil(np.log2(n_experts))), 1)
        self.gating_network = torch.nn.Linear(n_embd, n_experts, bias=False).to(torch.bfloat16).to("cuda")
        self.grouped_linear1 = fp8_gmm_ops.GroupedLinear(n_embd, n_inner, n_experts, dtype=torch.bfloat16)
        self.grouped_linear2 = fp8_gmm_ops.GroupedLinear(n_inner, n_embd, n_experts, dtype=torch.bfloat16)
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


# Bf16Module
input = torch.randn(16, 1024, dtype=torch.bfloat16, device="cuda", requires_grad=True)
model_bf16 = Bf16Module(1024, 768, 16, 2).to(torch.bfloat16).to("cuda")
out = model_bf16(input)
out.backward(torch.randn(*out.size(), dtype=torch.bfloat16, device="cuda"))
print(out)
print(input.grad)

# Fp8Module
input = torch.randn(16, 1024, dtype=torch.bfloat16, device="cuda", requires_grad=True)
model_fp8 = Fp8Module(1024, 768, 16, 2)
fp8_recipe = recipe.DelayedScaling(margin=0, interval=1, fp8_format=recipe.Format.HYBRID)
with tex.fp8_autocast(enabled=True, fp8_recipe=fp8_recipe):
    out = model_fp8(input)
out.backward(torch.randn(*out.size(), dtype=torch.bfloat16, device="cuda"))
print(out)
print(input.grad)
