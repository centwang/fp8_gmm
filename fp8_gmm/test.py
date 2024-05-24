import unittest

import torch
from parameterized import parameterized

from fp8_gmm import backend


class TestBackend(unittest.TestCase):
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

    def get_transpose_shapes(self, padded):
        trans_shapes = []
        for shape in TestBackend.shapes:
            trans_shape = [shape[1], shape[0]]
            if padded and shape[0] % 16 != 0:
                trans_shape[1] += 16 - shape[0] % 16
            trans_shapes.append(trans_shape)
        return trans_shapes

    def dgelu(self, x, dy):
        tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
        return (0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (1 + tanh_out)) * dy

    @parameterized.expand([torch.float8_e4m3fn, torch.float8_e5m2])
    def test_multi_quantize(self, dtype):
        inputs = [torch.randn(shape, dtype=torch.bfloat16, device="cuda") for shape in TestBackend.shapes]
        outputs = [torch.empty_like(input, dtype=dtype) for input in inputs]
        scales = [torch.randn(1, dtype=torch.float32, device="cuda") for _ in range(len(inputs))]
        amaxes = [torch.tensor([0.0], dtype=torch.float32, device="cuda") for _ in range(len(inputs))]
        backend.multi_quantize(inputs, outputs, scales, amaxes)
        ref_outputs = [input * scale for input, scale in zip(inputs, scales)]
        ref_amaxes = [input.abs().max().to(torch.float32) for input in inputs]
        for output, ref_output in zip(outputs, ref_outputs):
            self.assertTrue(torch.allclose(output.to(torch.float32), ref_output, atol=0.2, rtol=0.2))
        for amax, ref_amax in zip(amaxes, ref_amaxes):
            self.assertTrue(torch.allclose(amax, ref_amax, atol=0.05, rtol=0.05))

    @parameterized.expand(
        [
            (torch.float8_e4m3fn, True),
            (torch.float8_e4m3fn, False),
            (torch.float8_e5m2, True),
            (torch.float8_e5m2, False),
        ]
    )
    def test_multi_transpose(self, dtype, padded):
        trans_shapes = self.get_transpose_shapes(padded)
        inputs = [torch.randn(shape, dtype=torch.bfloat16, device="cuda").to(dtype) for shape in TestBackend.shapes]
        outputs = [torch.empty(trans_shape, dtype=dtype, device="cuda") for trans_shape in trans_shapes]
        backend.multi_transpose(inputs, outputs)
        ref_outputs = [input.t() for input in inputs]
        if padded:
            for idx, shape in enumerate(TestBackend.shapes):
                delta = trans_shapes[idx][1] - shape[0]
                if delta != 0:
                    ref_outputs[idx] = torch.nn.functional.pad(ref_outputs[idx], (0, delta))
        for output, ref_output in zip(outputs, ref_outputs):
            self.assertTrue(torch.allclose(output.to(torch.bfloat16), ref_output.to(torch.bfloat16)))

    @parameterized.expand(
        [
            (torch.float8_e4m3fn, True),
            (torch.float8_e4m3fn, False),
            (torch.float8_e5m2, True),
            (torch.float8_e5m2, False),
        ]
    )
    def test_multi_cast_transpose(self, dtype, padded):
        trans_shapes = self.get_transpose_shapes(padded)
        inputs = [torch.randn(shape, dtype=torch.bfloat16, device="cuda") for shape in TestBackend.shapes]
        outputs = [torch.empty_like(input, dtype=dtype) for input in inputs]
        trans_outputs = [torch.empty(trans_shape, dtype=dtype, device="cuda") for trans_shape in trans_shapes]
        scales = [torch.randn(1, dtype=torch.float32, device="cuda") for _ in range(len(inputs))]
        amaxes = [torch.tensor([0.0], dtype=torch.float32, device="cuda") for _ in range(len(inputs))]
        backend.multi_cast_transpose(inputs, outputs, trans_outputs, scales, amaxes)
        ref_outputs = [input * scale for input, scale in zip(inputs, scales)]
        ref_trans_outputs = [input.t() * scale for input, scale in zip(inputs, scales)]
        ref_amaxes = [input.abs().max().to(torch.float32) for input in inputs]
        if padded:
            for idx, shape in enumerate(TestBackend.shapes):
                delta = trans_shapes[idx][1] - shape[0]
                if delta != 0:
                    ref_trans_outputs[idx] = torch.nn.functional.pad(ref_trans_outputs[idx], (0, delta))
        for output, ref_output in zip(outputs, ref_outputs):
            self.assertTrue(torch.allclose(output.to(torch.float32), ref_output, atol=0.2, rtol=0.2))
        for trans_output, ref_trans_output in zip(trans_outputs, ref_trans_outputs):
            self.assertTrue(torch.allclose(trans_output.to(torch.float32), ref_trans_output, atol=0.2, rtol=0.2))
        for amax, ref_amax in zip(amaxes, ref_amaxes):
            self.assertTrue(torch.allclose(amax, ref_amax, atol=0.05, rtol=0.05))

    @parameterized.expand(
        [
            (torch.float8_e4m3fn, True),
            (torch.float8_e4m3fn, False),
            (torch.float8_e5m2, True),
            (torch.float8_e5m2, False),
        ]
    )
    def test_multi_cast_transpose_dgelu(self, dtype, padded):
        trans_shapes = self.get_transpose_shapes(padded)
        inputs = [torch.randn(shape, dtype=torch.bfloat16, device="cuda") for shape in TestBackend.shapes]
        gelu_inputs = [torch.randn(shape, dtype=torch.bfloat16, device="cuda") for shape in TestBackend.shapes]
        outputs = [torch.empty_like(input, dtype=dtype) for input in inputs]
        trans_outputs = [torch.empty(trans_shape, dtype=dtype, device="cuda") for trans_shape in trans_shapes]
        scales = [torch.randn(1, dtype=torch.float32, device="cuda") for _ in range(len(inputs))]
        amaxes = [torch.tensor([0.0], dtype=torch.float32, device="cuda") for _ in range(len(inputs))]
        backend.multi_cast_transpose_dgelu(inputs, gelu_inputs, outputs, trans_outputs, scales, amaxes)
        dgelu_outputs = [self.dgelu(gelu_input, input) for input, gelu_input in zip(inputs, gelu_inputs)]
        ref_outputs = [dgelu_output * scale for dgelu_output, scale in zip(dgelu_outputs, scales)]
        ref_trans_outputs = [dgelu_output.t() * scale for dgelu_output, scale in zip(dgelu_outputs, scales)]
        ref_amaxes = [dgelu_output.abs().max().to(torch.float32) for dgelu_output in dgelu_outputs]
        if padded:
            for idx, shape in enumerate(TestBackend.shapes):
                delta = trans_shapes[idx][1] - shape[0]
                if delta != 0:
                    ref_trans_outputs[idx] = torch.nn.functional.pad(ref_trans_outputs[idx], (0, delta))
        for output, ref_output in zip(outputs, ref_outputs):
            self.assertTrue(torch.allclose(output.to(torch.float32), ref_output, atol=0.2, rtol=0.2))
        for trans_output, ref_trans_output in zip(trans_outputs, ref_trans_outputs):
            self.assertTrue(torch.allclose(trans_output.to(torch.float32), ref_trans_output, atol=0.2, rtol=0.2))
        for amax, ref_amax in zip(amaxes, ref_amaxes):
            self.assertTrue(torch.allclose(amax, ref_amax, atol=0.05, rtol=0.05))

    @parameterized.expand(
        [
            ([32, 64], torch.float8_e4m3fn, [32, 64], torch.float8_e4m3fn, torch.bfloat16, False),
            ([64, 128], torch.float8_e5m2, [64, 128], torch.float8_e4m3fn, torch.bfloat16, True),
            ([32, 64], torch.float8_e4m3fn, [32, 64], torch.float8_e4m3fn, torch.float8_e4m3fn, False),
        ]
    )
    def test_cublas_fp8_gemm(self, a_shape, a_dtype, b_shape, b_dtype, c_dtype, backward):
        a_bf16 = torch.randn(a_shape, dtype=torch.bfloat16, device="cuda")
        b_bf16 = torch.randn(b_shape, dtype=torch.bfloat16, device="cuda")
        a = a_bf16.to(a_dtype)
        b = b_bf16.to(b_dtype)
        c = torch.empty(a.shape[0], b.shape[0], device=a.device, dtype=c_dtype)
        a_scale_inv = torch.randn(1, dtype=torch.float32, device="cuda")
        b_scale_inv = torch.randn(1, dtype=torch.float32, device="cuda")
        pre_gelu_out = None
        c_scale = None
        c_amax = None
        if c_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            pre_gelu_out = torch.empty(a.shape[0], b.shape[0], device=a.device, dtype=torch.bfloat16)
            c_scale = torch.randn(1, dtype=torch.float32, device="cuda")
            c_amax = torch.tensor([0.0], dtype=torch.float32, device="cuda")
        backend.cublas_fp8_gemm(a, b, c, a_scale_inv, b_scale_inv, pre_gelu_out, c_scale, c_amax, backward)
        ref_c = torch.matmul(a_bf16 * a_scale_inv, b_bf16.t() * b_scale_inv)
        if c_scale is not None:
            ref_pre_gelu_out = ref_c
            ref_c = torch.nn.functional.gelu(ref_c)
            ref_c_amax = ref_c.abs().max()
            ref_c = ref_c * c_scale
        self.assertTrue(torch.allclose(c.to(torch.float32), ref_c, atol=2, rtol=2))
        if c_scale is not None:
            self.assertTrue(torch.allclose(pre_gelu_out.to(torch.float32), ref_pre_gelu_out, atol=2, rtol=2))
            self.assertTrue(torch.allclose(c_amax, ref_c_amax, atol=0.05, rtol=0.05))

    @parameterized.expand(
        [
            (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.bfloat16, [16, 32, 16], 32, 64, False, False),
            (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.bfloat16, [16, 32, 16], 32, 64, False, True),
            (torch.float8_e5m2, torch.float8_e4m3fn, torch.bfloat16, [16, 32, 16], 32, 64, True, False),
            (torch.float8_e4m3fn, torch.float8_e4m3fn, torch.float8_e4m3fn, [16, 32, 16], 32, 64, False, False),
        ]
    )
    def test_fp8_gmm(self, a_dtype, b_dtype, c_dtype, group_sizes, k, n, backward, cutlass):
        num_groups = len(group_sizes)
        m = sum(group_sizes)
        a_bf16 = torch.randn(m, k, dtype=torch.bfloat16, device="cuda")
        b_bf16 = torch.randn(num_groups, n, k, dtype=torch.bfloat16, device="cuda")
        a = a_bf16.to(a_dtype)
        b = b_bf16.to(b_dtype)
        c = torch.empty(m, n, device=a.device, dtype=c_dtype)
        a_scale_invs = [torch.randn(1, dtype=torch.float32, device="cuda") for _ in range(num_groups)]
        b_scale_invs = [torch.randn(1, dtype=torch.float32, device="cuda") for _ in range(num_groups)]
        pre_gelu_out = None
        c_scales = None
        c_amaxes = None
        if c_dtype in [torch.float8_e4m3fn, torch.float8_e5m2]:
            pre_gelu_out = torch.empty(m, n, device=a.device, dtype=torch.bfloat16)
            c_scales = [torch.randn(1, dtype=torch.float32, device="cuda") for _ in range(num_groups)]
            c_amaxes = [torch.tensor([0.0], dtype=torch.float32, device="cuda") for _ in range(num_groups)]
        backend.fp8_gmm(
            a,
            b,
            torch.tensor(group_sizes),
            a_scale_invs,
            b_scale_invs,
            c,
            c_dtype,
            pre_gelu_out,
            c_scales,
            c_amaxes,
            backward,
            cutlass,
        )
        ref_c = torch.empty(m, n, device=a.device, dtype=torch.float32)
        start = 0
        for i in range(num_groups):
            end = start + group_sizes[i]
            torch.matmul(a_bf16[start:end] * a_scale_invs[i], b_bf16[i].t() * b_scale_invs[i], out=ref_c[start:end])
            start = end
        if c_scales is not None:
            ref_pre_gelu_out = ref_c
            ref_c = torch.nn.functional.gelu(ref_c)
            ref_amaxes = []
            start = 0
            for i in range(num_groups):
                end = start + group_sizes[i]
                ref_amaxes.append(ref_c[start:end].abs().max())
                torch.mul(ref_c[start:end], c_scales[i], out=ref_c[start:end])
                start = end
        self.assertTrue(torch.allclose(c.to(torch.float32), ref_c, atol=2, rtol=2))
        if c_scales is not None:
            self.assertTrue(torch.allclose(pre_gelu_out.to(torch.float32), ref_pre_gelu_out, atol=2, rtol=2))
            for c_amax, ref_c_amax in zip(c_amaxes, ref_amaxes):
                self.assertTrue(torch.allclose(c_amax, ref_c_amax, atol=0.05, rtol=0.05))


class TestOps(unittest.TestCase):
    def test_ops(self):
        print("test_ops")


if __name__ == "__main__":
    unittest.main()
