import math

import torch
from transformer_engine.pytorch.fp8 import get_fp8_te_dtype
from transformer_engine.pytorch.module.base import TransformerEngineBaseModule

from .backend import cublas_fp8_gemm, fp8_gmm, multi_cast_transpose, multi_quantize
from .utils import Fp8MetaWrapper, Fp8TensorType, MetaTensorType, cumsum_group_sizes, to_torch_dtype


class _GroupedLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, group_sizes, fp8_meta, is_grad_enabled, cutlass):
        need_padding = is_grad_enabled and weight.requires_grad
        num_groups, padded, cumsums, padded_cumsums = cumsum_group_sizes(
            group_sizes.tolist(), need_padding=need_padding
        )
        fp8_meta_wrapper = Fp8MetaWrapper(num_groups, fp8_meta)
        dtype = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=True)
        torch_dtype = to_torch_dtype(dtype)
        casts = []
        cast_fp8s = []
        cast_scales = []
        cast_amaxes = []
        cast_trans = []
        cast_trans_fp8s = []
        cast_trans_t_fp8s = []
        cast_trans_scales = []
        cast_trans_amaxes = []
        input_fp8 = torch.empty(*input.size(), dtype=torch_dtype, device=input.device)
        input_t_fp8 = None
        if is_grad_enabled and weight.requires_grad:
            cast_trans.extend(input[cumsums[i] : cumsums[i + 1]] for i in range(num_groups))
            cast_trans_fp8s.extend([input_fp8[cumsums[i] : cumsums[i + 1]] for i in range(num_groups)])
            if padded:
                input_t_fp8 = torch.empty(padded_cumsums[-1], input.size(1), dtype=torch_dtype, device=input.device)
                cast_trans_t_fp8s.extend(
                    [
                        input_t_fp8[padded_cumsums[i] : padded_cumsums[i + 1]].view(
                            -1, padded_cumsums[i + 1] - padded_cumsums[i]
                        )
                        for i in range(num_groups)
                    ]
                )
            else:
                input_t_fp8 = torch.empty_like(input_fp8)
                cast_trans_t_fp8s.extend(
                    [
                        input_t_fp8[cumsums[i] : cumsums[i + 1]].view(-1, cumsums[i + 1] - cumsums[i])
                        for i in range(num_groups)
                    ]
                )
            cast_trans_scales.extend(
                fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kInputGroup1, MetaTensorType.kScale)
            )
            cast_trans_amaxes.extend(
                fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kInputGroup1, MetaTensorType.kAmax)
            )
        else:
            casts.extend(input[cumsums[i] : cumsums[i + 1]] for i in range(num_groups))
            cast_fp8s.extend([input_fp8[cumsums[i] : cumsums[i + 1]] for i in range(num_groups)])
            cast_scales.extend(fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kInputGroup1, MetaTensorType.kScale))
            cast_amaxes.extend(fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kInputGroup1, MetaTensorType.kAmax))
        weight_fp8 = torch.empty(*weight.size(), dtype=torch_dtype, device=weight.device)
        weight_t_fp8 = None
        if is_grad_enabled and input.requires_grad:
            cast_trans.extend(weight[i] for i in range(num_groups))
            cast_trans_fp8s.extend([weight_fp8[i] for i in range(num_groups)])
            # Assume the sizes for MatMul can be devided by 16 so that cublasLtMatmul can be used.
            weight_t_fp8 = torch.empty(
                num_groups, weight.size(2), weight.size(1), dtype=torch_dtype, device=weight.device
            )
            cast_trans_t_fp8s.extend([weight_t_fp8[i] for i in range(num_groups)])
            cast_trans_scales.extend(
                fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kWeightGroup1, MetaTensorType.kScale)
            )
            cast_trans_amaxes.extend(
                fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kWeightGroup1, MetaTensorType.kAmax)
            )
        else:
            casts.extend(weight[i] for i in range(num_groups))
            cast_fp8s.extend([weight_fp8[i] for i in range(num_groups)])
            cast_scales.extend(fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kWeightGroup1, MetaTensorType.kScale))
            cast_amaxes.extend(fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kWeightGroup1, MetaTensorType.kAmax))
        if len(casts) > 0:
            multi_quantize(casts, cast_fp8s, cast_scales, cast_amaxes)
        if len(cast_trans) > 0:
            multi_cast_transpose(cast_trans, cast_trans_fp8s, cast_trans_t_fp8s, cast_trans_scales, cast_trans_amaxes)
        out = fp8_gmm(
            input_fp8,
            weight_fp8,
            group_sizes,
            fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kInputGroup1, MetaTensorType.kScaleInv),
            fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kWeightGroup1, MetaTensorType.kScaleInv),
            backward=False,
            cutlass=cutlass,
        )
        if is_grad_enabled:
            ctx.save_for_backward(input_t_fp8, weight_t_fp8, group_sizes)
            ctx.num_groups = num_groups
            fp8_meta_wrapper.set_forward_scale_inv_cloned()
            ctx.fp8_meta_wrapper = fp8_meta_wrapper
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
        (input_t_fp8, weight_t_fp8, group_sizes) = ctx.saved_tensors
        num_groups = ctx.num_groups
        fp8_meta_wrapper = ctx.fp8_meta_wrapper
        padded = ctx.padded
        cumsums = ctx.cumsums
        padded_cumsums = ctx.padded_cumsums
        fp8_meta = ctx.fp8_meta
        cutlass = ctx.cutlass
        grad_out_dtype = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=False)
        torch_grad_out_dtype = to_torch_dtype(grad_out_dtype)
        grad_outs = [grad_out[cumsums[i] : cumsums[i + 1]] for i in range(num_groups)]
        grad_out_scales = fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kGradOutputGroup1, MetaTensorType.kScale)
        grad_out_amaxes = fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kGradOutputGroup1, MetaTensorType.kAmax)
        grad_out_fp8 = torch.empty(*grad_out.size(), dtype=torch_grad_out_dtype, device=grad_out.device)
        grad_out_fp8s = [grad_out_fp8[cumsums[i] : cumsums[i + 1]] for i in range(num_groups)]
        grad_out_t_fp8 = None
        grad_out_t_fp8s = []
        if ctx.weight_requires_grad:
            if padded:
                grad_out_t_fp8 = torch.empty(
                    padded_cumsums[-1], grad_out.size(1), dtype=torch_grad_out_dtype, device=grad_out.device
                )
                grad_out_t_fp8s = [
                    grad_out_t_fp8[padded_cumsums[i] : padded_cumsums[i + 1]].view(
                        -1, padded_cumsums[i + 1] - padded_cumsums[i]
                    )
                    for i in range(num_groups)
                ]
            else:
                grad_out_t_fp8 = torch.empty_like(grad_out_fp8)
                grad_out_t_fp8s = [
                    grad_out_t_fp8[cumsums[i] : cumsums[i + 1]].view(-1, cumsums[i + 1] - cumsums[i])
                    for i in range(num_groups)
                ]
            multi_cast_transpose(grad_outs, grad_out_fp8s, grad_out_t_fp8s, grad_out_scales, grad_out_amaxes)
        else:
            multi_quantize(grad_outs, grad_out_fp8s, grad_out_scales, grad_out_amaxes)
        grad_input = None
        if ctx.input_requires_grad:
            grad_input = fp8_gmm(
                grad_out_fp8,
                weight_t_fp8,
                group_sizes,
                fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kGradOutputGroup1, MetaTensorType.kScaleInv),
                fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kForwardWeightGroup1, MetaTensorType.kScaleInv),
                backward=True,
                cutlass=cutlass,
            )
        grad_weight = None
        if ctx.weight_requires_grad:
            grad_weight = torch.empty(*ctx.weight_shape, dtype=torch.bfloat16, device=weight_t_fp8.device)
            grad_out_scale_invs = fp8_meta_wrapper.get_meta_tensors(
                Fp8TensorType.kGradOutputGroup1, MetaTensorType.kScaleInv
            )
            forward_input_scale_invs = fp8_meta_wrapper.get_meta_tensors(
                Fp8TensorType.kForwardInputGroup1, MetaTensorType.kScaleInv
            )
            for i in range(num_groups):
                start, end = (padded_cumsums[i], padded_cumsums[i + 1]) if padded else (cumsums[i], cumsums[i + 1])
                cublas_fp8_gemm(
                    grad_out_t_fp8s[i],
                    input_t_fp8[start:end].view(-1, end - start),
                    grad_weight[i],
                    grad_out_scale_invs[i],
                    forward_input_scale_invs[i],
                    backward=True,
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
        torch.nn.init.uniform_(self.weight_tensor.data, -math.sqrt(1.0 / in_features), math.sqrt(1.0 / in_features))
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
            args += [input, self.weight_tensor, group_sizes, self.fp8_meta, torch.is_grad_enabled(), self.cutlass]
            return fn(*args)
