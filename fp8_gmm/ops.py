import math

import torch
from transformer_engine.pytorch import cpp_extensions as tex
from transformer_engine.pytorch.fp8 import get_fp8_te_dtype
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule, get_workspace

from .backend import fp8_gmm


def to_torch_dtype(dtype):
    if dtype == tex.DType.kFloat8E4M3:
        return torch.float8_e4m3fn
    elif dtype == tex.DType.kFloat8E5M2:
        return torch.float8_e5m2
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


class _GroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, group_sizes, fp8_meta, is_grad_enabled):
        num_groups = weight.size(0)
        group_sizes_tensor = torch.tensor(group_sizes)
        cumsum_group_sizes = [0]
        for i in range(num_groups):
            cumsum_group_sizes.append(cumsum_group_sizes[-1] + group_sizes[i])
        dtype = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
        torch_dtype = to_torch_dtype(dtype)
        input_fp8 = torch.empty(*input.size(), dtype=torch_dtype, device=input.device)
        weight_fp8 = torch.empty(*weight.size(), dtype=torch_dtype, device=weight.device)
        weight_t_fp8 = None
        if input.requires_grad:
            weight_t_fp8 = torch.empty(
                num_groups,
                weight.size(2),
                weight.size(1),
                dtype=torch_dtype,
                device=weight.device,
            )
        scale = fp8_meta["scaling_fwd"].scale
        scale_inv = fp8_meta["scaling_fwd"].scale_inv
        amax_history = fp8_meta["scaling_fwd"].amax_history
        weight_groups = []
        weight_scales = []
        weight_fp8_groups = []
        weight_t_fp8_groups = []
        weight_amax_historys = []
        weight_scale_invs = []
        # TODO: write kernels to fuse multi cast and multi mul respectively.
        for i in range(num_groups):
            start, end = cumsum_group_sizes[i], cumsum_group_sizes[i + 1]
            input_group = input[start:end]
            input_fp8_group = input_fp8[start:end]
            tex.cast_to_fp8_noalloc(
                input_group,
                scale[i * 3],
                input_fp8_group,
                amax_history[0][i * 3],
                scale_inv[i * 3],
                dtype,
            )
            if is_grad_enabled and input.requires_grad:
                weight_groups.append(weight[i])
                weight_scales.append(scale[i * 3 + 1])
                weight_fp8_groups.append(weight_fp8[i])
                weight_t_fp8_groups.append(weight_t_fp8[i])
                weight_amax_historys.append(amax_history[0][i * 3 + 1])
                weight_scale_invs.append(scale_inv[i * 3 + 1])
            else:
                weight_group = weight[i]
                weight_fp8_group = weight_fp8[i]
                tex.cast_to_fp8_noalloc(
                    weight_group,
                    scale[i * 3 + 1],
                    weight_fp8_group,
                    amax_history[0][i * 3 + 1],
                    scale_inv[i * 3 + 1],
                    dtype,
                )
        if is_grad_enabled and input.requires_grad:
            tex.fused_multi_cast_transpose(
                weight_groups,
                weight_scales,
                weight_fp8_groups,
                weight_t_fp8_groups,
                weight_amax_historys,
                weight_scale_invs,
                dtype,
            )
        out = fp8_gmm(input_fp8, weight_fp8, group_sizes_tensor)
        for i in range(num_groups):
            out_group = out[cumsum_group_sizes[i] : cumsum_group_sizes[i + 1]]
            torch.mul(out_group, scale_inv[i * 3] * scale_inv[i * 3 + 1], out=out_group)
        if is_grad_enabled:
            ctx.save_for_backward(input_fp8, weight_t_fp8, scale_inv.clone(), group_sizes_tensor)
            ctx.num_groups = num_groups
            ctx.cumsum_group_sizes = cumsum_group_sizes
            ctx.fp8_meta = fp8_meta
            ctx.forward_dtype = dtype
            ctx.weight_shape = weight.size()
            ctx.input_requires_grad = input.requires_grad
            ctx.weight_requires_grad = weight.requires_grad
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (input_fp8, weight_t_fp8, fw_scale_inv, group_sizes_tensor) = ctx.saved_tensors
        num_groups = ctx.num_groups
        cumsum_group_sizes = ctx.cumsum_group_sizes
        fp8_meta = ctx.fp8_meta
        grad_out_dtype = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=False)
        torch_grad_out_dtype = to_torch_dtype(grad_out_dtype)
        grad_out_fp8 = torch.empty(*grad_out.size(), dtype=torch_grad_out_dtype, device=grad_out.device)
        scale = fp8_meta["scaling_bwd"].scale
        scale_inv = fp8_meta["scaling_bwd"].scale_inv
        amax_history = fp8_meta["scaling_bwd"].amax_history
        for i in range(num_groups):
            start, end = cumsum_group_sizes[i], cumsum_group_sizes[i + 1]
            grad_out_group = grad_out[start:end]
            grad_out_fp8_group = grad_out_fp8[start:end]
            tex.cast_to_fp8_noalloc(
                grad_out_group,
                scale[i * 2],
                grad_out_fp8_group,
                amax_history[0][i * 2],
                scale_inv[i * 2],
                grad_out_dtype,
            )
        grad_input = None
        if ctx.input_requires_grad:
            grad_input = fp8_gmm(grad_out_fp8, weight_t_fp8, group_sizes_tensor)
            for i in range(num_groups):
                grad_input_group = grad_input[cumsum_group_sizes[i] : cumsum_group_sizes[i + 1]]
                torch.mul(
                    grad_input_group,
                    scale_inv[i * 2] * fw_scale_inv[i * 3 + 1],
                    out=grad_input_group,
                )
        grad_weight = None
        if ctx.weight_requires_grad:
            grad_weight = torch.empty_like(
                *ctx.weight_shape,
                dtype=torch.bfloat16,
                device=weight_t_fp8.device,
            )
            workspace = get_workspace()
            for i in range(num_groups):
                start, end = cumsum_group_sizes[i], cumsum_group_sizes[i + 1]
                input_fp8_group = input_fp8[start:end]
                grad_out_fp8_group = grad_out_fp8[start:end]
                grad_weight_group = grad_weight[i]
                tex.te_gemm(
                    input_fp8_group,
                    fw_scale_inv[i * 3],
                    ctx.forward_dtype,
                    False,
                    grad_out_fp8_group,
                    scale_inv[i * 2],
                    grad_out_dtype,
                    True,
                    grad_weight_group,
                    grad_out_dtype,
                    scale[i * 2 + 1],
                    tex.DType.kBFloat16,
                    amax_history[0][i * 2 + 1],
                    torch.Tensor(),
                    tex.DType.kBFloat16,
                    torch.Tensor(),
                    False,
                    workspace,
                    workspace.shape[0],
                    False,
                    True,
                )
        return (grad_input, grad_weight, None, None, None)


class GroupedLinear(TransformerEngineBaseModule):
    def __init__(self, in_features, out_features, num_groups, dtype, device="cuda"):
        super().__init__()
        # Support bfloat16 only for now.
        assert dtype == torch.bfloat16
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.weight_tensor = torch.empty(num_groups, out_features, in_features, dtype=dtype, device=device)
        torch.nn.init.uniform_(
            self.weight_tensor.data,
            -math.sqrt(1.0 / in_features),
            math.sqrt(1.0 / in_features),
        )
        self.register_parameter("weight", torch.nn.Parameter(self.weight_tensor))

    def get_fp8_weights_scratchpad(self, is_first_microbatch):
        assert is_first_microbatch is None
        return [None, None]

    def forward(self, input, group_sizes, is_first_microbatch=None):
        with self.prepare_forward(input, is_first_microbatch, self.num_groups):
            if torch.is_grad_enabled():
                fn = _GroupedLinear.apply
                args = []
            else:
                fn = _GroupedLinear.forward
                args = [None]
            args += [
                input,
                self.weight_tensor,
                group_sizes,
                self.fp8_meta,
                torch.is_grad_enabled(),
            ]
            return fn(*args)
