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
from .utils import Fp8MetaWrapper, Fp8TensorType, MetaTensorType, cumsum_group_sizes, to_torch_dtype


class _GroupedMlp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, fc1_weight, fc2_weight, group_sizes, fp8_meta, is_grad_enabled, has_tp, tp_group):
        input_requires_grad = input.requires_grad
        weight_requires_grad = fc1_weight.requires_grad
        need_padding = is_grad_enabled and weight_requires_grad
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
        if is_grad_enabled and weight_requires_grad:
            cast_trans.extend([input[cumsums[i] : cumsums[i + 1]] for i in range(num_groups)])
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
            casts.extend([input[cumsums[i] : cumsums[i + 1]] for i in range(num_groups)])
            cast_fp8s.extend([input_fp8[cumsums[i] : cumsums[i + 1]] for i in range(num_groups)])
            cast_scales.extend(fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kInputGroup1, MetaTensorType.kScale))
            cast_amaxes.extend(fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kInputGroup1, MetaTensorType.kAmax))
        fc1_weight_fp8 = torch.empty(*fc1_weight.size(), dtype=torch_dtype, device=fc1_weight.device)
        fc1_weight_t_fp8 = None
        fc2_weight_fp8 = torch.empty(*fc2_weight.size(), dtype=torch_dtype, device=fc2_weight.device)
        fc2_weight_t_fp8 = None
        if is_grad_enabled and input_requires_grad:
            # Assume the sizes for MatMul can be devided by 16 so that cublasLtMatmul can be used.
            fc1_weight_t_fp8 = torch.empty(
                num_groups, fc1_weight.size(2), fc1_weight.size(1), dtype=torch_dtype, device=fc1_weight.device
            )
            fc2_weight_t_fp8 = torch.empty(
                num_groups, fc2_weight.size(2), fc2_weight.size(1), dtype=torch_dtype, device=fc2_weight.device
            )
            cast_trans.extend([fc1_weight[i] for i in range(num_groups)])
            cast_trans.extend([fc2_weight[i] for i in range(num_groups)])
            cast_trans_fp8s.extend([fc1_weight_fp8[i] for i in range(num_groups)])
            cast_trans_fp8s.extend([fc2_weight_fp8[i] for i in range(num_groups)])
            cast_trans_t_fp8s.extend([fc1_weight_t_fp8[i] for i in range(num_groups)])
            cast_trans_t_fp8s.extend([fc2_weight_t_fp8[i] for i in range(num_groups)])
            cast_trans_scales.extend(
                fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kWeightGroup1, MetaTensorType.kScale)
            )
            cast_trans_scales.extend(
                fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kWeightGroup2, MetaTensorType.kScale)
            )
            cast_trans_amaxes.extend(
                fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kWeightGroup1, MetaTensorType.kAmax)
            )
            cast_trans_amaxes.extend(
                fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kWeightGroup2, MetaTensorType.kAmax)
            )
        else:
            casts.extend([fc1_weight[i] for i in range(num_groups)])
            casts.extend([fc2_weight[i] for i in range(num_groups)])
            cast_fp8s.extend([fc1_weight_fp8[i] for i in range(num_groups)])
            cast_fp8s.extend([fc2_weight_fp8[i] for i in range(num_groups)])
            cast_scales.extend(fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kWeightGroup1, MetaTensorType.kScale))
            cast_scales.extend(fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kWeightGroup2, MetaTensorType.kScale))
            cast_amaxes.extend(fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kWeightGroup1, MetaTensorType.kAmax))
            cast_amaxes.extend(fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kWeightGroup2, MetaTensorType.kAmax))
        if len(casts) > 0:
            multi_quantize(casts, cast_fp8s, cast_scales, cast_amaxes)
        if len(cast_trans) > 0:
            multi_cast_transpose(cast_trans, cast_trans_fp8s, cast_trans_t_fp8s, cast_trans_scales, cast_trans_amaxes)
        pre_gelu_out = torch.empty(input.size(0), fc1_weight.size(1), dtype=torch.bfloat16, device=input.device)
        gelu_out_fp8 = torch.empty(input.size(0), fc1_weight.size(1), dtype=torch_dtype, device=input.device)
        # fc1 gmm.
        fp8_gmm(
            input_fp8,
            fc1_weight_fp8,
            group_sizes,
            fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kInputGroup1, MetaTensorType.kScaleInv),
            fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kWeightGroup1, MetaTensorType.kScaleInv),
            c=gelu_out_fp8,
            pre_gelu_out=pre_gelu_out,
            c_scales=fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kInputGroup2, MetaTensorType.kScale),
            c_amaxes=fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kInputGroup2, MetaTensorType.kAmax),
            backward=False,
            cutlass=False,  # Cultass kernel doesn't have Gelu fusion.
        )
        if not is_grad_enabled:
            del pre_gelu_out
        # fc2 gmm.
        out = fp8_gmm(
            gelu_out_fp8,
            fc2_weight_fp8,
            group_sizes,
            fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kInputGroup2, MetaTensorType.kScaleInv),
            fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kWeightGroup2, MetaTensorType.kScaleInv),
            backward=False,
            cutlass=False,
        )
        if not (is_grad_enabled and input_requires_grad):
            del gelu_out_fp8
            gelu_out_fp8 = None
        if is_grad_enabled:
            ctx.save_for_backward(
                input_t_fp8, fc1_weight_t_fp8, fc2_weight_t_fp8, pre_gelu_out, gelu_out_fp8, group_sizes
            )
            ctx.num_groups = num_groups
            fp8_meta_wrapper.set_forward_scale_inv_cloned()
            ctx.fp8_meta_wrapper = fp8_meta_wrapper
            ctx.padded = padded
            ctx.cumsums = cumsums
            ctx.padded_cumsums = padded_cumsums
            ctx.fp8_meta = fp8_meta
            ctx.forward_dtype = dtype
            ctx.fc1_weight_shape = fc1_weight.size()
            ctx.fc2_weight_shape = fc2_weight.size()
            ctx.input_requires_grad = input_requires_grad
            ctx.weight_requires_grad = weight_requires_grad
            ctx.has_tp = has_tp
            if has_tp:
                assert tp_group is not None
            ctx.tp_group = tp_group
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (input_t_fp8, fc1_weight_t_fp8, fc2_weight_t_fp8, pre_gelu_out, gelu_out_fp8, group_sizes) = ctx.saved_tensors
        num_groups = ctx.num_groups
        fp8_meta_wrapper = ctx.fp8_meta_wrapper
        padded = ctx.padded
        cumsums = ctx.cumsums
        padded_cumsums = ctx.padded_cumsums
        fp8_meta = ctx.fp8_meta
        grad_out_dtype = get_fp8_te_dtype(fp8_meta["recipe"], fprop_tensor=False)
        torch_grad_out_dtype = to_torch_dtype(grad_out_dtype)
        grad_out_fp8 = torch.empty(*grad_out.size(), dtype=torch_grad_out_dtype, device=grad_out.device)
        grad_outs = [grad_out[cumsums[i] : cumsums[i + 1]] for i in range(num_groups)]
        grad_out_fp8s = [grad_out_fp8[cumsums[i] : cumsums[i + 1]] for i in range(num_groups)]
        grad_out_scales = fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kGradOutputGroup1, MetaTensorType.kScale)
        grad_out_amaxes = fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kGradOutputGroup1, MetaTensorType.kAmax)
        grad_out_t_fp8 = None
        grad_out_t_fp8s = []
        gelu_out_t_fp8 = None
        gelu_out_t_fp8s = []
        if ctx.weight_requires_grad:
            gelu_out_fp8s = [gelu_out_fp8[cumsums[i] : cumsums[i + 1]] for i in range(num_groups)]
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
                gelu_out_t_fp8 = torch.empty(
                    padded_cumsums[-1], gelu_out_fp8.size(1), dtype=gelu_out_fp8.dtype, device=grad_out.device
                )
                gelu_out_t_fp8s = [
                    gelu_out_t_fp8[padded_cumsums[i] : padded_cumsums[i + 1]].view(
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
                gelu_out_t_fp8 = torch.empty_like(gelu_out_fp8)
                gelu_out_t_fp8s = [
                    gelu_out_t_fp8[cumsums[i] : cumsums[i + 1]].view(-1, cumsums[i + 1] - cumsums[i])
                    for i in range(num_groups)
                ]
            multi_cast_transpose(grad_outs, grad_out_fp8s, grad_out_t_fp8s, grad_out_scales, grad_out_amaxes)
            multi_transpose(gelu_out_fp8s, gelu_out_t_fp8s)
        else:
            multi_quantize(grad_outs, grad_out_fp8s, grad_out_scales, grad_out_amaxes)
        grad_fc2_weight = None
        if ctx.weight_requires_grad:
            grad_fc2_weight = torch.empty(*ctx.fc2_weight_shape, dtype=torch.bfloat16, device=fc2_weight_t_fp8.device)
            grad_out_scale_invs = fp8_meta_wrapper.get_meta_tensors(
                Fp8TensorType.kGradOutputGroup1, MetaTensorType.kScaleInv
            )
            fw_gelu_out_scale_inv = fp8_meta_wrapper.get_meta_tensors(
                Fp8TensorType.kForwardInputGroup2, MetaTensorType.kScaleInv
            )
            for i in range(num_groups):
                cublas_fp8_gemm(
                    grad_out_t_fp8s[i],
                    gelu_out_t_fp8s[i],
                    grad_fc2_weight[i],
                    grad_out_scale_invs[i],
                    fw_gelu_out_scale_inv[i],
                    backward=True,
                )
        grad_gelu_out = fp8_gmm(
            grad_out_fp8,
            fc2_weight_t_fp8,
            group_sizes,
            fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kGradOutputGroup1, MetaTensorType.kScaleInv),
            fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kForwardWeightGroup2, MetaTensorType.kScaleInv),
            backward=True,
            cutlass=False,  # Cultass kernel doesn't have Gelu fusion.
        )
        # Always do cast, transpose and dgelu for now.
        # Maybe need a new kernel for cast_dgelu only when weight requires no grad.
        grad_gelu_outs = [grad_gelu_out[cumsums[i] : cumsums[i + 1]] for i in range(num_groups)]
        pre_gelu_outs = [pre_gelu_out[cumsums[i] : cumsums[i + 1]] for i in range(num_groups)]
        grad_pre_gelu_out_fp8 = torch.empty(
            *pre_gelu_out.size(), dtype=torch_grad_out_dtype, device=pre_gelu_out.device
        )
        grad_pre_gelu_out_fp8s = [grad_pre_gelu_out_fp8[cumsums[i] : cumsums[i + 1]] for i in range(num_groups)]
        grad_pre_gelu_out_t_fp8 = None
        grad_pre_gelu_out_t_fp8s = []
        if padded:
            grad_pre_gelu_out_t_fp8 = torch.empty(
                padded_cumsums[-1], pre_gelu_out.size(1), dtype=torch_grad_out_dtype, device=pre_gelu_out.device
            )
            grad_pre_gelu_out_t_fp8s = [
                grad_pre_gelu_out_t_fp8[padded_cumsums[i] : padded_cumsums[i + 1]].view(
                    -1, padded_cumsums[i + 1] - padded_cumsums[i]
                )
                for i in range(num_groups)
            ]
        else:
            grad_pre_gelu_out_t_fp8 = torch.empty_like(grad_pre_gelu_out_fp8)
            grad_pre_gelu_out_t_fp8s = [
                grad_pre_gelu_out_t_fp8[cumsums[i] : cumsums[i + 1]].view(-1, cumsums[i + 1] - cumsums[i])
                for i in range(num_groups)
            ]
        pre_gelu_out_scales = fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kGradOutputGroup2, MetaTensorType.kScale)
        pre_gelu_out_amaxes = fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kGradOutputGroup2, MetaTensorType.kAmax)
        multi_cast_transpose_dgelu(
            grad_gelu_outs,
            pre_gelu_outs,
            grad_pre_gelu_out_fp8s,
            grad_pre_gelu_out_t_fp8s,
            pre_gelu_out_scales,
            pre_gelu_out_amaxes,
        )
        grad_input = None
        handle = None
        if ctx.input_requires_grad:
            grad_input = fp8_gmm(
                grad_pre_gelu_out_fp8,
                fc1_weight_t_fp8,
                group_sizes,
                fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kGradOutputGroup2, MetaTensorType.kScaleInv),
                fp8_meta_wrapper.get_meta_tensors(Fp8TensorType.kForwardWeightGroup1, MetaTensorType.kScaleInv),
                backward=True,
                cutlass=False,  # Cultass kernel doesn't have Gelu fusion.
            )
            if ctx.has_tp:
                handle = torch.distributed.all_reduce(
                    grad_input, op=torch.distributed.ReduceOp.SUM, group=ctx.tp_group, async_op=True
                )
        grad_fc1_weight = None
        if ctx.weight_requires_grad:
            grad_fc1_weight = torch.empty(*ctx.fc1_weight_shape, dtype=torch.bfloat16, device=fc2_weight_t_fp8.device)
            grad_pre_gelu_out_scale_invs = fp8_meta_wrapper.get_meta_tensors(
                Fp8TensorType.kGradOutputGroup2, MetaTensorType.kScaleInv
            )
            fw_input_scale_inv = fp8_meta_wrapper.get_meta_tensors(
                Fp8TensorType.kForwardInputGroup1, MetaTensorType.kScaleInv
            )
            for i in range(num_groups):
                start, end = (padded_cumsums[i], padded_cumsums[i + 1]) if padded else (cumsums[i], cumsums[i + 1])
                cublas_fp8_gemm(
                    grad_pre_gelu_out_t_fp8s[i],
                    input_t_fp8[start:end].view(-1, end - start),
                    grad_fc1_weight[i],
                    grad_pre_gelu_out_scale_invs[i],
                    fw_input_scale_inv[i],
                    backward=True,
                )
        if handle is not None:
            handle.wait()
        return (grad_input, grad_fc1_weight, grad_fc2_weight, None, None, None, None, None)


class GroupedMlp(TransformerEngineBaseModule):
    def __init__(self, in_features, out_features, num_groups, dtype, device="cuda"):
        super().__init__()
        # Support bfloat16 only for now.
        assert dtype == torch.bfloat16
        self.num_groups = num_groups
        self.fc1_weight = torch.nn.Parameter(
            torch.empty(num_groups, out_features, in_features, dtype=dtype, device=device)
        )
        self.fc2_weight = torch.nn.Parameter(
            torch.empty(num_groups, in_features, out_features, dtype=dtype, device=device)
        )
        torch.nn.init.uniform_(self.fc1_weight.data, -math.sqrt(1.0 / in_features), math.sqrt(1.0 / in_features))
        torch.nn.init.uniform_(self.fc2_weight.data, -math.sqrt(1.0 / out_features), math.sqrt(1.0 / out_features))

    def get_fp8_weights_scratchpad(self, is_first_microbatch):
        assert is_first_microbatch is None
        return [None, None, None, None]

    def forward(self, input, group_sizes, has_tp=False, tp_group=None, is_first_microbatch=None):
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
                has_tp,
                tp_group,
            ]
            return fn(*args)
