# NOTE: Torch needs to be imported before the custom
# extensions. Otherwise libc10.so cannot be found.
import torch

# TODO(tgale): Wrap this in a try-block with better
# error message and instructions for building the
# c++ operations.
import fp8_gmm_backend as backend


def _fp8_allocate_output(a, b):
    return torch.empty(a.shape[0], b.shape[1], device=a.device, dtype=torch.bfloat16)

def fp8_gmm(a, b, batch_sizes, c=None):
    if c is None:
        c = _fp8_allocate_output(a, b)
    backend.fp8_gmm(a, b, c, batch_sizes)
    return c
