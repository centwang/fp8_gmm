import fp8_gmm_backend as backend
import torch


def _fp8_allocate_output(a, b):
    return torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=torch.bfloat16)


def fp8_gmm(a, b, batch_sizes, c=None):
    if c is None:
        c = _fp8_allocate_output(a, b)
    backend.fp8_gmm(a, b, c, batch_sizes)
    return c
