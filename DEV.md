We use MoE torch modules in [example.py](https://github.com/er3x3/fp8_gmm/blob/main/fp8_gmm/example.py) to compare the performance between BFloat16 run and Float8 run.
- Bf16Module is for BFloat16 run, which calls implementation from repo [grouped_gemm](https://github.com/tgale96/grouped_gemm) for the grouped GEMM compute.
- Fp8Module is for Float8 run, which uses GroupedLinear in this repo (a sub-class of TransformerEngineBaseModule) as implementation. We provides both Cutlass and cuBlas implementations for the grouped GEMM compute.

Running output of the example.py:
```
Bf16Module run: 20.887 ms
Fp8Module cublasLtMatmul run: 18.298 ms
Fp8Module cutlass run: 100.171 ms
```

# Cutlass
The Cutlass kernel is in [fp8_gmm.cu](https://github.com/er3x3/fp8_gmm/blob/main/csrc/fp8_gmm.cu). The template is from one of the examples from official Cutlass repo. There is no grouped GEMM kernel from Cublas for now, so the difference between Cutlass kernel and cuBlas kernel is, for each grouped GEMM, Cutlass kernel launches only one kernel, while cuBlas implementation will use FOR loop to launches multiple cuBlas GEMM kernels, each for a single group. Performance number above shows that the Cutlass kernel is very slow. Further performance profiling on H100 shows that even for BFloat16 run using grouped_gemm repo, cuBlas kernel is picked instead of Cutlass kernel. This means that for both BFloat16 and Float8, the Cutlass kernel is not well tuned on H100, so we will no longer use Cutlass kernel for our further Float8 training development.

![image](https://github.com/er3x3/fp8_gmm/assets/11661208/b789261e-10c6-44a8-809b-8a2947d2749f)

# cuBlas GEMM: BFloat16 vs. Float8
Below shows the performance profling results for cuBlas GEMM kernel (The last column shows the duration of each kernel).

|BFloat16|Float8|
|--------|------|
|![image](https://github.com/er3x3/fp8_gmm/assets/11661208/006f79b9-b630-4179-8dbc-316b8af9e56b)|![image](https://github.com/er3x3/fp8_gmm/assets/11661208/1dc524e6-345b-40e6-9cee-f6de979c6730)|

From the numbers we can see that Float8 kernel is 2x faster than BFloat16 kernel. So we expect Float8 running should be faster than BFloat16 running if the grouped GEMM compute is the major part during the training. <br/>

# Float8 GEMM
According to the documentation of [cublasLtMatmul](https://docs.nvidia.com/cuda/cublas/#cublasltmatmul), there are some limitations for Float8 GEMM compute which will affect our implementation for grouped GEMM:
- A must be transposed and B non-transposed (the "TN" format). Since cuBlas kernel is column-major on definition, for row-major, it actually requires [m, k] * [n, k]T to produces [m, n] result. For training, there is one GEMM for forward and 2 GEMMs for backward, which require both original and transposed data for all input, weight and output_grad. Besides casting data from BFLoat16 to Float8 for GEMM compute, we also need extra time to perform the permutation for all 3 big tensors.
- All matrix pointers must be 16-byte aligned. Since the input and output_grad tensors are contiguous and contain num_group sub-tensors, it actually requires K must can be devided by 16. But the group size for each group is determined by a gating Linear, which is changed step by step, and always cannot meet such requirement.

To solve above problems:
- Instead of lauching multiple kernels for casting and permutation for each sub-tensor, we will have a fused kernel to do all these jobs.
- The fused kernel will perform both casting and permutation, as well as computing Amax for each sub-tensor.
- The fused kernel will also pad sub-tensors with 0 so that the K for each sub-tensor can be devided by 16.

The implementation of this fused kernel is in [multi_pad_cast_transpose.cu](https://github.com/er3x3/fp8_gmm/blob/main/csrc/multi_pad_fusion.cu). Below shows the performance profling result (2 kernels are launched, one for aligned data input, which is the vectorized version, one for non-aligned data input).

![image](https://github.com/er3x3/fp8_gmm/assets/11661208/9526dddd-d9f4-4751-a574-8bc0747f2915)

End up we still need to spend extra time for data processing before Float8 GEMM compute, and since the tensors are big, this extra time is not small, we will not able to achieve 2x performance gain at the end. (From above, 20.887 ms vs. 18.298 ms for end-to-end run, the performance gain for only grouped BEMM is bigger than this end-to-end run comparison.)

# TODO
Current MoE module uses two GroupedLinears with a torch.nn.GELU between. cuBlas's FP8 GEMM supported GELU fusion, so we can fuse these two GroupedLinears and one GELU to a single GroupedMLP module, so that we will no need to launch GELU and its gradient kernels separately. By doing this, we can also remove some data processing (casting and permutation) between to GroupedLinears, for example, the output of the first GroupedLinear can be kept in Float8 format so that we won't cast it again for the 2nd GroupedLinear.
