import fp8_gmm_backend as backend
import torch
from transformer_engine.pytorch import cpp_extensions as tex
from transformer_engine.pytorch.module.base import get_workspace

from .utils import to_tex_dtype


def multi_quantize(inputs, outputs, scales, amaxes):
    backend.multi_quantize(inputs, outputs, scales, amaxes)


def multi_transpose(inputs, outputs):
    backend.multi_pad_transpose(inputs, outputs)


def multi_cast_transpose(inputs, outputs, trans_outputs, scales, amaxes):
    backend.multi_pad_cast_transpose(inputs, outputs, trans_outputs, scales, amaxes)


def multi_cast_transpose_dgelu(inputs, gelu_inputs, outputs, trans_outputs, scales, amaxes):
    backend.multi_pad_cast_transpose_dgelu(inputs, gelu_inputs, outputs, trans_outputs, scales, amaxes)


# For FP8 cublasLtMatmul, A must be transposed and B non-transposed (The “TN” format).
def cublas_fp8_gemm(a, b, c, a_scale_inv, b_scale_inv, pre_gelu_out=None, c_scale=None, c_amax=None, backward=False):
    c_dtype = to_tex_dtype(c.dtype)
    if c_dtype in [tex.DType.kFloat8E4M3, tex.DType.kFloat8E5M2]:
        assert c_scale is not None and c_amax is not None
    empty_tensor = torch.Tensor()
    workspace = get_workspace()
    tex.te_gemm(
        b,
        b_scale_inv,
        to_tex_dtype(b.dtype),
        True,
        a,
        a_scale_inv,
        to_tex_dtype(a.dtype),
        False,
        c,
        empty_tensor if c_scale is None else c_scale,
        c_dtype,
        empty_tensor if c_amax is None else c_amax,
        empty_tensor,
        tex.DType.kBFloat16,
        empty_tensor if pre_gelu_out is None else pre_gelu_out,
        False,
        workspace,
        workspace.shape[0],
        False,
        backward,
        0,
    )


# For FP8, both cublas and cutlass support only row-major layout for A and column-major layout for B.
def fp8_gmm(
    a,
    b,
    group_sizes,
    a_scale_invs,
    b_scale_invs,
    c=None,
    c_dtype=torch.bfloat16,
    pre_gelu_out=None,
    c_scales=None,
    c_amaxes=None,
    backward=False,
    cutlass=False,
):
    if c is None:
        c = torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=c_dtype)
    num_groups = len(group_sizes)
    cumsum_group_sizes = (
        torch.cat((torch.zeros(1, device=group_sizes.device, dtype=group_sizes.dtype), group_sizes)).cumsum(0).tolist()
    )
    if cutlass:
        assert c.dtype == torch.bfloat16 and pre_gelu_out is None
        backend.fp8_gmm(a, b, c, group_sizes)
        c_groups = [c[cumsum_group_sizes[i] : cumsum_group_sizes[i + 1]] for i in range(num_groups)]
        backend.multi_scale_mul(c_groups, a_scale_invs, b_scale_invs)
    else:
        for i in range(num_groups):
            start, end = cumsum_group_sizes[i], cumsum_group_sizes[i + 1]
            cublas_fp8_gemm(
                a[start:end],
                b[i],
                c[start:end],
                a_scale_invs[i],
                b_scale_invs[i],
                pre_gelu_out=pre_gelu_out[start:end] if pre_gelu_out is not None else None,
                c_scale=c_scales[i] if c_scales is not None else None,
                c_amax=c_amaxes[i] if c_amaxes is not None else None,
                backward=backward,
            )
    return c
