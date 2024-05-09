from enum import IntEnum

import torch
from transformer_engine.pytorch import cpp_extensions as tex


class MetaTensorType(IntEnum):
    kScale = 0
    kScaleInv = 1
    kAmax = 2


class Fp8TensorType(IntEnum):
    kInputGroup1 = 0
    kInputGroup2 = 1
    kWeightGroup1 = 2
    kWeightGroup2 = 3
    kGradOutputGroup1 = 4
    kGradOutputGroup2 = 5
    kForwardInputGroup1 = 6
    kForwardInputGroup2 = 7
    kForwardWeightGroup1 = 8
    kForwardWeightGroup2 = 9


def to_tex_dtype(dtype):
    if dtype == torch.float8_e4m3fn:
        return tex.DType.kFloat8E4M3
    elif dtype == torch.float8_e5m2:
        return tex.DType.kFloat8E5M2
    elif dtype == torch.bfloat16:
        return tex.DType.kBFloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def to_torch_dtype(dtype):
    if dtype == tex.DType.kFloat8E4M3:
        return torch.float8_e4m3fn
    elif dtype == tex.DType.kFloat8E5M2:
        return torch.float8_e5m2
    elif dtype == tex.DType.kBFloat16:
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def cumsum_group_sizes(group_sizes, need_padding=False):
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


class Fp8MetaWrapper:
    _FORWARD_OFFSET = 3
    _BACKWARD_OFFSET = 2

    def __init__(self, num_groups, fp8_meta):
        self._num_groups = num_groups
        self._fp8_meta = fp8_meta
        self._forward_scale_inv_cloned = None

    def num_groups(self):
        return self._num_groups

    def set_forward_scale_inv_cloned(self):
        self._forward_scale_inv_cloned = self._fp8_meta["scaling_fwd"].scale_inv.clone()

    def get_meta_tensors(self, fp8_tensor_type, meta_tensor_type):
        meta_tensor = None
        if fp8_tensor_type >= Fp8TensorType.kForwardInputGroup1:
            assert meta_tensor_type == MetaTensorType.kScaleInv
            meta_tensor = self._forward_scale_inv_cloned
        else:
            meta = (
                self._fp8_meta["scaling_fwd"]
                if fp8_tensor_type < Fp8TensorType.kGradOutputGroup1
                else self._fp8_meta["scaling_bwd"]
            )
            if meta_tensor_type == MetaTensorType.kScale:
                meta_tensor = meta.scale
            elif meta_tensor_type == MetaTensorType.kScaleInv:
                meta_tensor = meta.scale_inv
            elif meta_tensor_type == MetaTensorType.kAmax:
                meta_tensor = meta.amax_history[0]
        assert meta_tensor is not None
        group_offset = 0 if "Group1" in fp8_tensor_type.name else self._num_groups
        stage_offset = (
            Fp8MetaWrapper._BACKWARD_OFFSET if "GradOutput" in fp8_tensor_type.name else Fp8MetaWrapper._FORWARD_OFFSET
        )
        type_offset = 1 if "Weight" in fp8_tensor_type.name else 0
        return [meta_tensor[(group_offset + i) * stage_offset + type_offset] for i in range(self._num_groups)]
