import math

import torch
from transformer_engine.pytorch.fp8 import get_fp8_te_dtype
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule

from .backend import (
    cublas_fp8_gemm,
    fp8_gmm,
    multi_cast_transpose,
    multi_cast_transpose_dgelu,
    multi_quantize,
    multi_transpose,
)
from .utils import Fp8MetaOffset, cumsum_group_sizes, to_torch_dtype

"""
cast_trans(input) -> input_fp8, input_t_fp8
cast_trans(fc1) -> fc1_fp8, fc1_t_fp8
input_fp8[t, h] * fc1_fp8[g, i, h](T) -> gelu_out_fp8[t, i], fc1_out[t, i]
cast_trans(fc2) -> fc2_fp8, fc2_t_fp8
gelu_out_fp8[t, i] * fc2_fp8[g, h, i](T) -> out[t, h]

Grad:
cast_trans(out_grad) -> out_grad_fp8, out_grad_t_fp8
trans(gelu_out_fp8) -> gelu_out_t_fp8
g(out_grad_t_fp8[h, t] * gelu_out_t_fp8[i, t](T)) -> fc2_grad[g, h, i]
out_grad_fp8[t, h] * fc2_t_fp8[g, i, h](T) -> gelu_out_grad[t, i]
cast_trans_dgelu(gelu_out_grad, fc1_out) -> fc1_out_fp8, fp1_out_t_fp8
g(fp1_out_t_fp8[i, t] * input_t_fp8[h, t](T)) -> fc1_grad[g, i, h]
fc1_out_fp8[t, i] * fc1_t_fp8[g, h, i](T) -> input_grad[t, h]
"""


class _GroupedMlp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, fc1_weight, fc2_weight, group_sizes, fp8_meta, is_grad_enabled):
        input_requires_grad = input.requires_grad
        weight_requires_grad = fc1_weight.requires_grad
        need_padding = is_grad_enabled and weight_requires_grad
        num_groups, padded, cumsums, padded_cumsums = cumsum_group_sizes(
            group_sizes.tolist(), need_padding=need_padding
        )
        fp8_meta_offset = Fp8MetaOffset(num_groups)
        dtype = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
        torch_dtype = to_torch_dtype(dtype)
        input_fp8 = torch.empty(*input.size(), dtype=torch_dtype, device=input.device)
        input_t_fp8 = None
        if is_grad_enabled and weight_requires_grad:
            if padded:
                input_t_fp8 = torch.empty(padded_cumsums[-1], input.size(1), dtype=torch_dtype, device=input.device)
            else:
                input_t_fp8 = torch.empty_like(input_fp8)
        fc1_weight_fp8 = torch.empty(*fc1_weight.size(), dtype=torch_dtype, device=fc1_weight.device)
        fc1_weight_t_fp8 = None
        if is_grad_enabled and input_requires_grad:
            # Assume the sizes for MatMul can be devided by 16 so that cublasLtMatmul can be used.
            fc1_weight_t_fp8 = torch.empty(
                num_groups, fc1_weight.size(2), fc1_weight.size(1), dtype=torch_dtype, device=fc1_weight.device
            )
        fc2_weight_fp8 = torch.empty(*fc2_weight.size(), dtype=torch_dtype, device=fc2_weight.device)
        fc2_weight_t_fp8 = None
        if is_grad_enabled and input_requires_grad:
            # Assume the sizes for MatMul can be devided by 16 so that cublasLtMatmul can be used.
            fc2_weight_t_fp8 = torch.empty(
                num_groups, fc2_weight.size(2), fc2_weight.size(1), dtype=torch_dtype, device=fc2_weight.device
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
            input_offset = fp8_meta_offset.input_group1(i)
            fc1_offset = fp8_meta_offset.weight_group1(i)
            fc2_offset = fp8_meta_offset.weight_group2(i)
            if is_grad_enabled and weight_requires_grad:
                cast_trans.append(input[start:end])
                cast_trans_fp8s.append(input_fp8[start:end])
                if padded:
                    padded_start, padded_end = padded_cumsums[i], padded_cumsums[i + 1]
                    cast_trans_t_fp8s.append(input_t_fp8[padded_start:padded_end].view(-1, padded_end - padded_start))
                else:
                    cast_trans_t_fp8s.append(input_t_fp8[start:end].view(-1, end - start))
                cast_trans_scales.append(scale[input_offset])
                cast_trans_amaxes.append(amax_history[0][input_offset])
                cast_trans_scale_invs.append(scale_inv[input_offset])
            else:
                casts.append(input[start:end])
                cast_fp8s.append(input_fp8[start:end])
                cast_scales.append(scale[input_offset])
                cast_amaxes.append(amax_history[0][input_offset])
            if is_grad_enabled and input_requires_grad:
                cast_trans.append(fc1_weight[i])
                cast_trans_fp8s.append(fc1_weight_fp8[i])
                cast_trans_t_fp8s.append(fc1_weight_t_fp8[i])
                cast_trans_scales.append(scale[fc1_offset])
                cast_trans_amaxes.append(amax_history[0][fc1_offset])
                cast_trans_scale_invs.append(scale_inv[fc1_offset])
                cast_trans.append(fc2_weight[i])
                cast_trans_fp8s.append(fc2_weight_fp8[i])
                cast_trans_t_fp8s.append(fc2_weight_t_fp8[i])
                cast_trans_scales.append(scale[fc2_offset])
                cast_trans_amaxes.append(amax_history[0][fc2_offset])
                cast_trans_scale_invs.append(scale_inv[fc2_offset])
            else:
                casts.append(fc1_weight[i])
                cast_fp8s.append(fc1_weight_fp8[i])
                cast_scales.append(scale[fc1_offset])
                cast_amaxes.append(amax_history[0][fc1_offset])
                casts.append(fc2_weight[i])
                cast_fp8s.append(fc2_weight_fp8[i])
                cast_scales.append(scale[fc2_offset])
                cast_amaxes.append(amax_history[0][fc2_offset])
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
        pre_gelu_out = torch.empty(input.size(0), fc1_weight.size(1), dtype=torch.bfloat16, device=input.device)
        gelu_out_fp8 = torch.empty(input.size(0), fc1_weight.size(1), dtype=torch_dtype, device=input.device)
        # fc1 gmm.
        fp8_gmm(
            input_fp8,
            fc1_weight_fp8,
            group_sizes,
            [scale_inv[fp8_meta_offset.input_group1(i)] for i in range(num_groups)],
            [scale_inv[fp8_meta_offset.weight_group1(i)] for i in range(num_groups)],
            c=gelu_out_fp8,
            pre_gelu_out=pre_gelu_out,
            c_scales=[scale[fp8_meta_offset.input_group2(i)] for i in range(num_groups)],
            c_amaxes=[amax_history[0][fp8_meta_offset.input_group2(i)] for i in range(num_groups)],
            backward=False,
            cutlass=False,  # Cultass kernel doesn't have Gelu fusion.
        )
        # fc2 gmm.
        out = fp8_gmm(
            gelu_out_fp8,
            fc2_weight_fp8,
            group_sizes,
            [scale_inv[fp8_meta_offset.input_group2(i)] for i in range(num_groups)],
            [scale_inv[fp8_meta_offset.weight_group2(i)] for i in range(num_groups)],
            backward=False,
            cutlass=False,
        )
        if is_grad_enabled:
            ctx.save_for_backward(
                input_t_fp8,
                fc1_weight_t_fp8,
                fc2_weight_t_fp8,
                pre_gelu_out,
                gelu_out_fp8,
                scale_inv.clone(),
                group_sizes,
            )
            ctx.num_groups = num_groups
            ctx.fp8_meta_offset = fp8_meta_offset
            ctx.padded = padded
            ctx.cumsums = cumsums
            ctx.padded_cumsums = padded_cumsums
            ctx.fp8_meta = fp8_meta
            ctx.forward_dtype = dtype
            ctx.fc1_weight_shape = fc1_weight.size()
            ctx.fc2_weight_shape = fc2_weight.size()
            ctx.input_requires_grad = input_requires_grad
            ctx.weight_requires_grad = weight_requires_grad
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (input_t_fp8, fc1_weight_t_fp8, fc2_weight_t_fp8, pre_gelu_out, gelu_out_fp8, fw_scale_inv, group_sizes) = (
            ctx.saved_tensors
        )
        num_groups = ctx.num_groups
        fp8_meta_offset = ctx.fp8_meta_offset
        padded = ctx.padded
        cumsums = ctx.cumsums
        padded_cumsums = ctx.padded_cumsums
        fp8_meta = ctx.fp8_meta
        grad_out_dtype = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=False)
        torch_grad_out_dtype = to_torch_dtype(grad_out_dtype)
        grad_out_fp8 = torch.empty(*grad_out.size(), dtype=torch_grad_out_dtype, device=grad_out.device)
        grad_out_t_fp8 = None
        gelu_out_t_fp8 = None
        if ctx.weight_requires_grad:
            if padded:
                grad_out_t_fp8 = torch.empty(
                    padded_cumsums[-1], grad_out.size(1), dtype=torch_grad_out_dtype, device=grad_out.device
                )
                gelu_out_t_fp8 = torch.empty(
                    padded_cumsums[-1], gelu_out_fp8.size(1), dtype=gelu_out_fp8.dtype, device=grad_out.device
                )
            else:
                grad_out_t_fp8 = torch.empty_like(grad_out_fp8)
                gelu_out_t_fp8 = torch.empty_like(gelu_out_fp8)
        scale = fp8_meta["scaling_bwd"].scale
        scale_inv = fp8_meta["scaling_bwd"].scale_inv
        amax_history = fp8_meta["scaling_bwd"].amax_history
        grad_outs = []
        grad_out_fp8s = []
        grad_out_t_fp8s = []
        grad_out_scales = []
        grad_out_amaxes = []
        grad_out_scale_invs = []
        gelu_out_fp8s = []
        gelu_out_t_fp8s = []
        for i in range(num_groups):
            start, end = cumsums[i], cumsums[i + 1]
            grad_outs.append(grad_out[start:end])
            grad_out_fp8s.append(grad_out_fp8[start:end])
            grad_out_scales.append(scale[fp8_meta_offset.output_grad_group1(i)])
            grad_out_amaxes.append(amax_history[0][fp8_meta_offset.output_grad_group1(i)])
            if ctx.weight_requires_grad:
                gelu_out_fp8s.append(gelu_out_fp8[start:end])
                if padded:
                    padded_start, padded_end = padded_cumsums[i], padded_cumsums[i + 1]
                    grad_out_t_fp8s.append(grad_out_t_fp8[padded_start:padded_end].view(-1, padded_end - padded_start))
                    gelu_out_t_fp8s.append(gelu_out_t_fp8[padded_start:padded_end].view(-1, padded_end - padded_start))
                else:
                    grad_out_t_fp8s.append(grad_out_t_fp8[start:end].view(-1, end - start))
                    gelu_out_t_fp8s.append(gelu_out_t_fp8[start:end].view(-1, end - start))
                grad_out_scale_invs.append(scale_inv[fp8_meta_offset.output_grad_group1(i)])
        if ctx.weight_requires_grad:
            multi_cast_transpose(
                grad_outs, grad_out_fp8s, grad_out_t_fp8s, grad_out_scales, grad_out_amaxes, grad_out_scale_invs, padded
            )
            multi_transpose(gelu_out_fp8s, gelu_out_t_fp8s)
        else:
            multi_quantize(grad_outs, grad_out_fp8s, grad_out_scales, grad_out_amaxes)
        grad_gelu_out = None
        if ctx.input_requires_grad:
            grad_gelu_out = fp8_gmm(
                grad_out_fp8,
                fc2_weight_t_fp8,
                group_sizes,
                [scale_inv[fp8_meta_offset.output_grad_group1(i)] for i in range(num_groups)],
                [fw_scale_inv[fp8_meta_offset.weight_group2(i)] for i in range(num_groups)],
                backward=True,
                cutlass=False,  # Cultass kernel doesn't have Gelu fusion.
            )
        grad_fc2_weight = None
        if ctx.weight_requires_grad:
            grad_fc2_weight = torch.empty(
                *ctx.fc2_weight_shape,
                dtype=torch.bfloat16,
                device=fc2_weight_t_fp8.device,
            )
            for i in range(num_groups):
                a = grad_out_t_fp8s[i]
                b = gelu_out_t_fp8s[i]
                cublas_fp8_gemm(
                    a,
                    b,
                    grad_fc2_weight[i],
                    grad_out_scale_invs[i],
                    fw_scale_inv[fp8_meta_offset.input_group2(i)],
                    backward=True,
                )
        grad_pre_gelu_out_fp8 = None
        grad_pre_gelu_out_t_fp8 = None
        grad_gelu_outs = []
        gelu_inputs = []
        grad_pre_gelu_out_fp8s = []
        grad_pre_gelu_out_t_fp8s = []
        grad_out_scales = []
        grad_out_amaxes = []
        for i in range(num_groups):
            start, end = cumsums[i], cumsums[i + 1]
            grad_gelu_outs.append(grad_gelu_out[start:end])
            grad_pre_gelu_out_fp8s.append(grad_pre_gelu_out_fp8[start:end])
            grad_out_scales.append(scale[fp8_meta_offset.output_grad_group2(i)])
            grad_out_amaxes.append(amax_history[0][fp8_meta_offset.output_grad_group2(i)])
            if padded:
                padded_start, padded_end = padded_cumsums[i], padded_cumsums[i + 1]
                grad_pre_gelu_out_t_fp8.append(
                    grad_out_t_fp8[padded_start:padded_end].view(-1, padded_end - padded_start)
                )
            else:
                grad_pre_gelu_out_t_fp8.append(grad_out_t_fp8[start:end].view(-1, end - start))
        multi_cast_transpose_dgelu(
            grad_gelu_outs,
            gelu_inputs,
            grad_pre_gelu_out_fp8s,
            grad_pre_gelu_out_t_fp8s,
            grad_out_scales,
            grad_out_amaxes,
        )
        grad_input = None
        if ctx.input_requires_grad:
            grad_input = fp8_gmm(
                grad_pre_gelu_out_fp8,
                fc1_weight_t_fp8,
                group_sizes,
                [scale_inv[fp8_meta_offset.output_grad_group2(i)] for i in range(num_groups)],
                [fw_scale_inv[fp8_meta_offset.weight_group1(i)] for i in range(num_groups)],
                backward=True,
                cutlass=False,  # Cultass kernel doesn't have Gelu fusion.
            )
        grad_fc1_weight = None
        if ctx.weight_requires_grad:
            grad_fc1_weight = torch.empty(
                *ctx.fc1_weight_shape,
                dtype=torch.bfloat16,
                device=fc2_weight_t_fp8.device,
            )
            for i in range(num_groups):
                start, end = (padded_cumsums[i], padded_cumsums[i + 1]) if padded else (cumsums[i], cumsums[i + 1])
                a = grad_pre_gelu_out_t_fp8s[i]
                b = input_t_fp8[start:end].view(-1, end - start)
                cublas_fp8_gemm(
                    a,
                    b,
                    grad_fc1_weight[i],
                    scale_inv[fp8_meta_offset.output_grad_group2],
                    fw_scale_inv[fp8_meta_offset.input_group1(i)],
                    backward=True,
                )
        return (grad_input, grad_fc1_weight, grad_fc2_weight, None, None, None)


class GroupedMlp(TransformerEngineBaseModule):
    def __init__(self, in_features, out_features, num_groups, dtype, device="cuda"):
        super().__init__()
        # Support bfloat16 only for now.
        assert dtype == torch.bfloat16
        self.in_features = in_features
        self.out_features = out_features
        self.num_groups = num_groups
        self.fc1_weight = torch.nn.Parameter(
            torch.empty(num_groups, out_features, in_features, dtype=dtype, device=device)
        )
        self.fc2_weight = torch.nn.Parameter(
            torch.empty(num_groups, in_features, out_features, dtype=dtype, device=device)
        )
        torch.nn.init.uniform_(
            self.fc1_weight.data,
            -math.sqrt(1.0 / in_features),
            math.sqrt(1.0 / in_features),
        )
        torch.nn.init.uniform_(
            self.fc2_weight.data,
            -math.sqrt(1.0 / out_features),
            math.sqrt(1.0 / out_features),
        )

    def get_fp8_weights_scratchpad(self, is_first_microbatch):
        assert is_first_microbatch is None
        return [None, None, None, None]

    def forward(self, input, group_sizes, is_first_microbatch=None):
        assert 0 not in group_sizes
        with self.prepare_forward(input, is_first_microbatch, self.num_groups * 2):
            if torch.is_grad_enabled():
                fn = _GroupedMlp.apply
                args = []
            else:
                fn = _GroupedMlp.forward
                args = [None]
            args += [
                input,
                self.fc1_weight,
                self.fc2_weight,
                group_sizes,
                self.fp8_meta,
                torch.is_grad_enabled(),
            ]
            return fn(*args)
