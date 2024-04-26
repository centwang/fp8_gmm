import math

import torch
from transformer_engine.pytorch import cpp_extensions as tex
from transformer_engine.pytorch.fp8 import get_fp8_te_dtype
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule, get_workspace

from .backend import cublas_fp8_gemm, fp8_gmm, multi_cast_transpose, multi_quantize

_META_FORWARD_OFFSET = 3
_META_BACKWARD_OFFSET = 2


def _meta_forward_input_offset(idx):
    return idx * _META_FORWARD_OFFSET


def _meta_forward_weight_offset(idx):
    return idx * _META_FORWARD_OFFSET + 1


def _meta_backward_grad_out_offset(idx):
    return idx * _META_BACKWARD_OFFSET


def _to_torch_dtype(dtype):
    if dtype == tex.DType.kFloat8E4M3:
        return torch.float8_e4m3fn
    elif dtype == tex.DType.kFloat8E5M2:
        return torch.float8_e5m2
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def _cumsum(group_sizes, need_padding=False):
    num_groups = len(group_sizes)
    padded = False
    cumsums = [0]
    padded_cumsums = [0] if need_padding else None
    for i in range(num_groups):
        cumsums.append(cumsums[-1] + group_sizes[i])
        if need_padding:
            group_size = group_sizes[i]
            remaining = group_size % 16
            if remaining > 0:
                padded = True
                group_size += 16 - remaining
            padded_cumsums.append(padded_cumsums[-1] + group_size)
    return num_groups, padded, cumsums, padded_cumsums


class _GroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, group_sizes, fp8_meta, is_grad_enabled, cutlass):
        need_padding = is_grad_enabled and weight.requires_grad
        num_groups, padded, cumsums, padded_cumsums = _cumsum(group_sizes.tolist(), need_padding=need_padding)
        dtype = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
        torch_dtype = _to_torch_dtype(dtype)
        input_fp8 = torch.empty(*input.size(), dtype=torch_dtype, device=input.device)
        input_t_fp8 = None
        if is_grad_enabled and weight.requires_grad:
            if padded:
                input_t_fp8 = torch.empty(padded_cumsums[-1], input.size(1), dtype=torch_dtype, device=input.device)
            else:
                input_t_fp8 = torch.empty_like(input_fp8)
        weight_fp8 = torch.empty(*weight.size(), dtype=torch_dtype, device=weight.device)
        weight_t_fp8 = None
        if is_grad_enabled and input.requires_grad:
            # Assume the sizes for MatMul can be devided by 16 so that cublasLtMatmul can be used.
            weight_t_fp8 = torch.empty(
                num_groups, weight.size(2), weight.size(1), dtype=torch_dtype, device=weight.device
            )
        scale = fp8_meta["scaling_fwd"].scale
        scale_inv = fp8_meta["scaling_fwd"].scale_inv
        amax_history = fp8_meta["scaling_fwd"].amax_history
        casts = []
        cast_fp8s = []
        cast_scales = []
        cast_amaxes = []
        cast_trans = []
        cast_trans_fp8s = []
        cast_trans_t_fp8s = []
        cast_trans_scales = []
        cast_trans_amaxes = []
        cast_trans_scale_invs = []
        for i in range(num_groups):
            start, end = cumsums[i], cumsums[i + 1]
            if is_grad_enabled and weight.requires_grad:
                cast_trans.append(input[start:end])
                cast_trans_fp8s.append(input_fp8[start:end])
                if padded:
                    padded_start, padded_end = padded_cumsums[i], padded_cumsums[i + 1]
                    cast_trans_t_fp8s.append(input_t_fp8[padded_start:padded_end].view(-1, padded_end - padded_start))
                else:
                    cast_trans_t_fp8s.append(input_t_fp8[start:end].view(-1, end - start))
                cast_trans_scales.append(scale[_meta_forward_input_offset(i)])
                cast_trans_amaxes.append(amax_history[0][_meta_forward_input_offset(i)])
                cast_trans_scale_invs.append(scale_inv[_meta_forward_input_offset(i)])
            else:
                casts.append(input[start:end])
                cast_fp8s.append(input_fp8[start:end])
                cast_scales.append(scale[_meta_forward_input_offset(i)])
                cast_amaxes.append(amax_history[0][_meta_forward_input_offset(i)])
            if is_grad_enabled and input.requires_grad:
                cast_trans.append(weight[i])
                cast_trans_fp8s.append(weight_fp8[i])
                cast_trans_t_fp8s.append(weight_t_fp8[i])
                cast_trans_scales.append(scale[_meta_forward_weight_offset(i)])
                cast_trans_amaxes.append(amax_history[0][_meta_forward_weight_offset(i)])
                cast_trans_scale_invs.append(scale_inv[_meta_forward_weight_offset(i)])
            else:
                casts.append(weight[i])
                cast_fp8s.append(weight_fp8[i])
                cast_scales.append(scale[_meta_forward_weight_offset(i)])
                cast_amaxes.append(amax_history[0][_meta_forward_weight_offset(i)])
        if len(casts) > 0:
            multi_quantize(casts, cast_fp8s, cast_scales, cast_amaxes)
        if len(cast_trans) > 0:
            multi_cast_transpose(
                cast_trans,
                cast_trans_fp8s,
                cast_trans_t_fp8s,
                cast_trans_scales,
                cast_trans_amaxes,
                cast_trans_scale_invs,
                padded,
            )
        out = fp8_gmm(
            input_fp8,
            weight_fp8,
            group_sizes,
            [scale_inv[_meta_forward_input_offset(i)] for i in range(num_groups)],
            [scale_inv[_meta_forward_weight_offset(i)] for i in range(num_groups)],
            c=None,
            cutlass=cutlass,
            backward=False,
        )
        if is_grad_enabled:
            ctx.save_for_backward(input_t_fp8, weight_t_fp8, scale_inv.clone(), group_sizes)
            ctx.num_groups = num_groups
            ctx.padded = padded
            ctx.cumsums = cumsums
            ctx.padded_cumsums = padded_cumsums
            ctx.fp8_meta = fp8_meta
            ctx.forward_dtype = dtype
            ctx.weight_shape = weight.size()
            ctx.input_requires_grad = input.requires_grad
            ctx.weight_requires_grad = weight.requires_grad
            ctx.cutlass = cutlass
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (input_t_fp8, weight_t_fp8, fw_scale_inv, group_sizes) = ctx.saved_tensors
        num_groups = ctx.num_groups
        padded = ctx.padded
        cumsums = ctx.cumsums
        padded_cumsums = ctx.padded_cumsums
        fp8_meta = ctx.fp8_meta
        cutlass = ctx.cutlass
        grad_out_dtype = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=False)
        torch_grad_out_dtype = _to_torch_dtype(grad_out_dtype)
        grad_out_fp8 = torch.empty(*grad_out.size(), dtype=torch_grad_out_dtype, device=grad_out.device)
        grad_out_t_fp8 = None
        if ctx.weight_requires_grad:
            if padded:
                grad_out_t_fp8 = torch.empty(
                    padded_cumsums[-1], grad_out.size(1), dtype=torch_grad_out_dtype, device=grad_out.device
                )
            else:
                grad_out_t_fp8 = torch.empty_like(grad_out_fp8)
        scale = fp8_meta["scaling_bwd"].scale
        scale_inv = fp8_meta["scaling_bwd"].scale_inv
        amax_history = fp8_meta["scaling_bwd"].amax_history
        grad_outs = []
        grad_out_fp8s = []
        grad_out_t_fp8s = []
        grad_out_scales = []
        grad_out_amaxes = []
        grad_out_scale_invs = []
        for i in range(num_groups):
            start, end = cumsums[i], cumsums[i + 1]
            grad_outs.append(grad_out[start:end])
            grad_out_fp8s.append(grad_out_fp8[start:end])
            grad_out_scales.append(scale[_meta_backward_grad_out_offset(i)])
            grad_out_amaxes.append(amax_history[0][_meta_backward_grad_out_offset(i)])
            if ctx.weight_requires_grad:
                if padded:
                    padded_start, padded_end = padded_cumsums[i], padded_cumsums[i + 1]
                    grad_out_t_fp8s.append(grad_out_t_fp8[padded_start:padded_end].view(-1, padded_end - padded_start))
                else:
                    grad_out_t_fp8s.append(grad_out_t_fp8[start:end].view(-1, end - start))
                grad_out_scale_invs.append(scale_inv[_meta_backward_grad_out_offset(i)])
        if ctx.weight_requires_grad:
            multi_cast_transpose(
                grad_outs, grad_out_fp8s, grad_out_t_fp8s, grad_out_scales, grad_out_amaxes, grad_out_scale_invs, padded
            )
        else:
            multi_quantize(grad_outs, grad_out_fp8s, grad_out_scales, grad_out_amaxes)
        grad_input = None
        if ctx.input_requires_grad:
            grad_input = fp8_gmm(
                grad_out_fp8,
                weight_t_fp8,
                group_sizes,
                [scale_inv[_meta_backward_grad_out_offset(i)] for i in range(num_groups)],
                [fw_scale_inv[_meta_forward_weight_offset(i)] for i in range(num_groups)],
                c=None,
                cutlass=cutlass,
                backward=True,
            )
        grad_weight = None
        if ctx.weight_requires_grad:
            grad_weight = torch.empty(
                *ctx.weight_shape,
                dtype=torch.bfloat16,
                device=weight_t_fp8.device,
            )
            for i in range(num_groups):
                start, end = (padded_cumsums[i], padded_cumsums[i + 1]) if padded else (cumsums[i], cumsums[i + 1])
                a = grad_out_t_fp8s[i]
                b = input_t_fp8[start:end].view(-1, end - start)
                cublas_fp8_gemm(
                    a,
                    b,
                    grad_weight[i],
                    grad_out_scale_invs[i],
                    fw_scale_inv[_meta_forward_input_offset(i)],
                    True,
                )
        return (grad_input, grad_weight, None, None, None, None)


class GroupedLinear(TransformerEngineBaseModule):
    def __init__(self, in_features, out_features, num_groups, dtype, device="cuda", cutlass=False):
        super().__init__()
        # Support bfloat16 only for now.
        assert dtype == torch.bfloat16
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.weight_tensor = torch.nn.Parameter(
            torch.empty(num_groups, out_features, in_features, dtype=dtype, device=device)
        )
        torch.nn.init.uniform_(
            self.weight_tensor.data,
            -math.sqrt(1.0 / in_features),
            math.sqrt(1.0 / in_features),
        )
        self.cutlass = cutlass

    def get_fp8_weights_scratchpad(self, is_first_microbatch):
        assert is_first_microbatch is None
        return [None, None]

    def forward(self, input, group_sizes, is_first_microbatch=None):
        assert 0 not in group_sizes
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
                self.cutlass,
            ]
            return fn(*args)
