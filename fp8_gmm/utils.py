import torch
from transformer_engine.pytorch import cpp_extensions as tex


def to_tex_dtype(dtype):
    if dtype == torch.float8_e4m3fn:
        return tex.DType.kFloat8E4M3
    elif dtype == torch.float8_e5m2:
        return tex.DType.kFloat8E5M2
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")


def to_torch_dtype(dtype):
    if dtype == tex.DType.kFloat8E4M3:
        return torch.float8_e4m3fn
    elif dtype == tex.DType.kFloat8E5M2:
        return torch.float8_e5m2
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

    def input_group1_scales(self):
        return [
            self._fp8_meta["scaling_fwd"].scale[i * Fp8MetaWrapper._FROWARD_OFFSET] for i in range(self._num_groups)
        ]

    def input_group1_scale_invs(self):
        return [
            self._fp8_meta["scaling_fwd"].scale_inv[i * Fp8MetaWrapper._FROWARD_OFFSET] for i in range(self._num_groups)
        ]

    def input_group1_amaxes(self):
        return [
            self._fp8_meta["scaling_fwd"].amax_history[i * Fp8MetaWrapper._FROWARD_OFFSET]
            for i in range(self._num_groups)
        ]

    def weight_group1_scales(self):
        return [
            self._fp8_meta["scaling_fwd"].scale[i * Fp8MetaWrapper._FROWARD_OFFSET + 1] for i in range(self._num_groups)
        ]

    def weight_group1_scale_invs(self):
        return [
            self._fp8_meta["scaling_fwd"].scale_inv[i * Fp8MetaWrapper._FROWARD_OFFSET + 1]
            for i in range(self._num_groups)
        ]

    def weight_group1_amaxes(self):
        return [
            self._fp8_meta["scaling_fwd"].amax_history[i * Fp8MetaWrapper._FROWARD_OFFSET + 1]
            for i in range(self._num_groups)
        ]

    def input_group2_scales(self):
        return [
            self._fp8_meta["scaling_fwd"].scale[(self._num_groups + i) * Fp8MetaWrapper._FROWARD_OFFSET]
            for i in range(self._num_groups)
        ]

    def input_group2_scale_invs(self):
        return [
            self._fp8_meta["scaling_fwd"].scale_inv[(self._num_groups + i) * Fp8MetaWrapper._FROWARD_OFFSET]
            for i in range(self._num_groups)
        ]

    def input_group2_amaxes(self):
        return [
            self._fp8_meta["scaling_fwd"].amax_history[(self._num_groups + i) * Fp8MetaWrapper._FROWARD_OFFSET]
            for i in range(self._num_groups)
        ]

    def weight_group2_scales(self):
        return [
            self._fp8_meta["scaling_fwd"].scale[(self._num_groups + i) * Fp8MetaWrapper._FROWARD_OFFSET + 1]
            for i in range(self._num_groups)
        ]

    def weight_group2_scale_invs(self):
        return [
            self._fp8_meta["scaling_fwd"].scale_inv[(self._num_groups + i) * Fp8MetaWrapper._FROWARD_OFFSET + 1]
            for i in range(self._num_groups)
        ]

    def weight_group2_amaxes(self):
        return [
            self._fp8_meta["scaling_fwd"].amax_history[(self._num_groups + i) * Fp8MetaWrapper._FROWARD_OFFSET + 1]
            for i in range(self._num_groups)
        ]

    def grad_output_group1_scales(self):
        return [
            self._fp8_meta["scaling_bwd"].scale[i * Fp8MetaWrapper._BACKWARD_OFFSET] for i in range(self._num_groups)
        ]

    def grad_output_group1_scale_invs(self):
        return [
            self._fp8_meta["scaling_bwd"].scale_inv[i * Fp8MetaWrapper._BACKWARD_OFFSET]
            for i in range(self._num_groups)
        ]

    def grad_output_group1_amaxes(self):
        return [
            self._fp8_meta["scaling_bwd"].amax_history[i * Fp8MetaWrapper._BACKWARD_OFFSET]
            for i in range(self._num_groups)
        ]

    def grad_output_group2_scales(self):
        return [
            self._fp8_meta["scaling_bwd"].scale[(self._num_groups + i) * Fp8MetaWrapper._BACKWARD_OFFSET]
            for i in range(self._num_groups)
        ]

    def grad_output_group2_scale_invs(self):
        return [
            self._fp8_meta["scaling_bwd"].scale_inv[(self._num_groups + i) * Fp8MetaWrapper._BACKWARD_OFFSET]
            for i in range(self._num_groups)
        ]

    def grad_output_group2_amaxes(self):
        return [
            self._fp8_meta["scaling_bwd"].amax_history[(self._num_groups + i) * Fp8MetaWrapper._BACKWARD_OFFSET]
            for i in range(self._num_groups)
        ]

    def forward_input_group1_scale_invs(self):
        return [self._forward_scale_inv_cloned[i * Fp8MetaWrapper._FROWARD_OFFSET] for i in range(self._num_groups)]

    def forward_input_group2_scale_invs(self):
        return [
            self._forward_scale_inv_cloned[(self._num_groups + i) * Fp8MetaWrapper._FROWARD_OFFSET]
            for i in range(self._num_groups)
        ]

    def forward_weight_group1_scale_invs(self):
        return [self._forward_scale_inv_cloned[i * Fp8MetaWrapper._FROWARD_OFFSET + 1] for i in range(self._num_groups)]

    def forward_weight_group2_scale_invs(self):
        return [
            self._forward_scale_inv_cloned[(self._num_groups + i) * Fp8MetaWrapper._FROWARD_OFFSET + 1]
            for i in range(self._num_groups)
        ]
