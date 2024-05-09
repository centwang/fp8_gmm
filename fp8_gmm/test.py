import unittest

import torch
from fp8_gmm import backend
from parameterized import parameterized


class TestBackend(unittest.TestCase):
    @parameterized.expand(
        [
            (torch.float8_e4m3fn, True),
            (torch.float8_e4m3fn, False),
            (torch.float8_e5m2, True),
            (torch.float8_e5m2, False),
        ]
    )
    def test_multi_pad_cast_transpose(self, dtype, padded):
        shapes = [
            (2048, 12288),
            (768, 1024),
            (256, 65536),
            (65536, 128),
            (256, 256),
            (120, 2080),
            (8, 8),
            (1, 3221),
            (2333, 1),
            (1481, 677),
        ]
        padded_shapes = []
        for shape in shapes:
            padded_shape = [shape[1], shape[0]]
            if padded and shape[0] % 16 != 0:
                padded_shape[1] += 16 - shape[0] % 16
            padded_shapes.append(padded_shape)
        inputs = [torch.randn(shape, dtype=torch.bfloat16, device="cuda") for shape in shapes]
        outputs = [torch.empty_like(input, dtype=dtype) for input in inputs]
        trans_outputs = [torch.empty(padded_shape, dtype=dtype, device="cuda") for padded_shape in padded_shapes]
        scales = [torch.randn(1, dtype=torch.float32, device="cuda") for _ in range(len(inputs))]
        amaxes = [torch.empty(1, dtype=torch.float32, device="cuda") for _ in range(len(inputs))]
        scale_invs = [torch.empty(1, dtype=torch.float32, device="cuda") for _ in range(len(inputs))]
        backend.multi_cast_transpose(inputs, outputs, trans_outputs, scales, amaxes, scale_invs, padded)
        ref_outputs = [input * scale for input, scale in zip(inputs, scales)]
        ref_trans_outputs = [input.t() * scale for input, scale in zip(inputs, scales)]
        ref_amaxes = [input.abs().max().to(torch.float32) for input in inputs]
        if padded:
            for idx, shape in enumerate(shapes):
                delta = padded_shapes[idx][1] - shape[0]
                if delta != 0:
                    ref_trans_outputs[idx] = torch.nn.functional.pad(ref_trans_outputs[idx], (0, delta))
        for output, ref_output in zip(outputs, ref_outputs):
            self.assertTrue(torch.allclose(output.to(torch.float32), ref_output, atol=0.2, rtol=0.2))
        for trans_output, ref_trans_output in zip(trans_outputs, ref_trans_outputs):
            self.assertTrue(torch.allclose(trans_output.to(torch.float32), ref_trans_output, atol=0.2, rtol=0.2))
        for amax, ref_amax in zip(amaxes, ref_amaxes):
            self.assertTrue(torch.allclose(amax, ref_amax, atol=0.05, rtol=0.05))


class TestOps(unittest.TestCase):
    def test_ops(self):
        print("test_ops")


if __name__ == "__main__":
    unittest.main()
