import fp8_gmm_backend as backend
import torch
from transformer_engine.pytorch import cpp_extensions as tex
from transformer_engine.pytorch.module.base import get_workspace


def _fp8_allocate_output(a, b):
    return torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=torch.bfloat16)


def _to_tex_dtype(dtype):
    if dtype == torch.float8_e4m3fn:
        return tex.DType.kFloat8E4M3
    elif dtype == torch.float8_e5m2:
        return tex.DType.kFloat8E5M2
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def multi_quantize(inputs, outputs, scales, amaxes):
    backend.multi_quantize(inputs, outputs, scales, amaxes)


def multi_cast_transpose(inputs, outputs, trans_outputs, scales, amaxes, scale_invs, padded):
    if padded:
        backend.multi_padded_cast_transpose(inputs, outputs, trans_outputs, scales, amaxes)
    else:
        tex.fused_multi_cast_transpose(
            inputs, scales, outputs, trans_outputs, amaxes, scale_invs, _to_tex_dtype(outputs[0].dtype)
        )


# For FP8 cublasLtMatmul, A must be transposed and B non-transposed (The “TN” format).
def cublas_fp8_gemm(a, b, c, a_scale_inv, b_scale_inv, backward):
    workspace = get_workspace()
    tex.te_gemm(
        b,
        b_scale_inv,
        _to_tex_dtype(b.dtype),
        True,
        a,
        a_scale_inv,
        _to_tex_dtype(a.dtype),
        False,
        c,
        torch.Tensor(),
        tex.DType.kBFloat16,
        torch.Tensor(),
        torch.Tensor(),
        tex.DType.kBFloat16,
        torch.Tensor(),
        False,
        workspace,
        workspace.shape[0],
        False,
        backward,
        0,
    )


# For FP8, both cublas and cutlass support only row-major layout for A and column-major layout for B.
def fp8_gmm(a, b, group_sizes, a_scale_invs, b_scale_invs, c=None, cutlass=True, backward=False):
    if c is None:
        c = _fp8_allocate_output(a, b)
    num_groups = len(group_sizes)
    cumsum_group_sizes = (
        torch.cat((torch.zeros(1, device=group_sizes.device, dtype=group_sizes.dtype), group_sizes)).cumsum(0).tolist()
    )
    if cutlass:
        backend.fp8_gmm(a, b, c, group_sizes)
        c_groups = [c[cumsum_group_sizes[i] : cumsum_group_sizes[i + 1]] for i in range(num_groups)]
        backend.multi_scale_mul(c_groups, a_scale_invs, b_scale_invs)
    else:
        for i in range(num_groups):
            cublas_fp8_gemm(
                a[cumsum_group_sizes[i] : cumsum_group_sizes[i + 1]],
                b[i],
                c[cumsum_group_sizes[i] : cumsum_group_sizes[i + 1]],
                a_scale_invs[i],
                b_scale_invs[i],
                backward,
            )
    return c
