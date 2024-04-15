#include "multi_pointwise.h"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace fp8_gmm {

namespace {

using fp32 = float;
using bf16 = nv_bfloat16;
using fp8e4m3 = __nv_fp8_e4m3;
using fp8e5m2 = __nv_fp8_e5m2;

constexpr int THREADS_PER_WARP = 32;

// Parameters to tune
constexpr int kThreadsPerBlock = 256;
constexpr int kNumElementsPerThread = 8;
constexpr int kNumElementsPerBlock = kThreadsPerBlock * kNumElementsPerThread;
constexpr int kMaxTensorsPerKernel = 64;
constexpr int kMaxBlocks = 65535;

template <typename T, int vec_size>
struct alignas(sizeof(T) * vec_size) aligned_vector {
  T val[vec_size];
};

struct MultiQuantizeArgs {
  void* input_list[kMaxTensorsPerKernel];
  void* output_list[kMaxTensorsPerKernel];
  void* scale_list[kMaxTensorsPerKernel];
  void* amax_list[kMaxTensorsPerKernel];
  int num_elements_list[kMaxTensorsPerKernel];
  int block_range[kMaxTensorsPerKernel + 1];
  int num_tensors;
};

struct MultiScaleMulArgs {
  void* input_list[kMaxTensorsPerKernel];
  void* scale_list_1[kMaxTensorsPerKernel];
  void* scale_list_2[kMaxTensorsPerKernel];
  int num_elements_list[kMaxTensorsPerKernel];
  int block_range[kMaxTensorsPerKernel + 1];
  int num_tensors;
};

template <int num_elems>
__device__ __forceinline__ float warp_reduce_max(const float m) {
  float tmp = m;
#pragma unroll
  for (int delta = num_elems / 2; delta > 0; delta /= 2) {
    const float other_m = __shfl_down_sync(0xFFFFFFFF, tmp, delta);
    __builtin_assume(tmp >= 0);
    __builtin_assume(other_m >= 0);
    tmp = fmaxf(tmp, other_m);
  }
  return tmp;
}

template <int num_warps, typename compute_t>
__device__ __forceinline__ compute_t reduce_max(const compute_t m, const int warpid) {
  __shared__ float staging[num_warps];
  constexpr int warp_size = 32;
  const float my_max = m;
  const float my_warp_max = warp_reduce_max<warp_size>(my_max);
  if (threadIdx.x % 32 == 0) {
    staging[warpid] = my_warp_max;
  }
  __syncthreads();
  compute_t result = 0;
  if (warpid == 0) {
    const float my_max = threadIdx.x < num_warps ? staging[threadIdx.x] : 0;
    result = warp_reduce_max<num_warps>(my_max);
  }
  return result;
}

__device__ __forceinline__ void atomicMaxFloat(float* addr, const float value) {
  atomicMax(reinterpret_cast<int*>(addr), __float_as_int(value));
}

template <bool aligned, typename OType>
__global__ void __launch_bounds__(kThreadsPerBlock) multi_quantize_kernel(MultiQuantizeArgs args) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  // Find tensor corresponding to block
  int tensor_id = 0;
  while (args.block_range[tensor_id + 1] <= bid) {
    ++tensor_id;
  }

  const bf16* input = reinterpret_cast<const bf16*>(args.input_list[tensor_id]);
  OType* output = reinterpret_cast<OType*>(args.output_list[tensor_id]);
  const fp32* scale_ptr = reinterpret_cast<fp32*>(args.scale_list[tensor_id]);
  const fp32 scale = scale_ptr == nullptr ? 1 : *scale_ptr;
  fp32* amax = reinterpret_cast<fp32*>(args.amax_list[tensor_id]);
  const int num_elements = args.num_elements_list[tensor_id];

  const int sub_bid = bid - args.block_range[tensor_id];
  const int warp_id = tid / THREADS_PER_WARP;

  using LoadT = aligned_vector<bf16, kNumElementsPerThread>;
  using StoreT = aligned_vector<OType, kNumElementsPerThread>;

  fp32 max = 0;
  bf16 src[kNumElementsPerThread];
  OType dst[kNumElementsPerThread];

  int id = sub_bid * kNumElementsPerBlock + tid * kNumElementsPerThread;
  if constexpr (aligned) {
    LoadT* value = reinterpret_cast<LoadT*>(&src);
    *value = *reinterpret_cast<const LoadT*>(&input[id]);
  } else {
    for (int i = 0; i < kNumElementsPerThread; ++i) {
      int j = id + i;
      if (j < num_elements) {
        src[i] = input[j];
      }
    }
  }
  for (int i = 0; i < kNumElementsPerThread; ++i) {
    fp32 val = __bfloat162float(src[i]);
    max = fmaxf(fabsf(val), max);
    dst[i] = static_cast<OType>(val * scale);
  }
  if constexpr (aligned) {
    StoreT* value = reinterpret_cast<StoreT*>(&dst);
    *reinterpret_cast<StoreT*>(&output[id]) = *value;
  } else {
    for (int i = 0; i < kNumElementsPerThread; ++i) {
      int j = id + i;
      if (j < num_elements) {
        output[j] = dst[i];
      }
    }
  }

  max = reduce_max<kThreadsPerBlock / THREADS_PER_WARP>(max, warp_id);
  if (tid == 0) {
    atomicMaxFloat(amax, max);
  }
}

template <bool aligned>
__global__ void __launch_bounds__(kThreadsPerBlock) multi_scale_mul_kernel(MultiScaleMulArgs args) {
  const int tid = threadIdx.x;
  const int bid = blockIdx.x;

  // Find tensor corresponding to block
  int tensor_id = 0;
  while (args.block_range[tensor_id + 1] <= bid) {
    ++tensor_id;
  }

  bf16* input = reinterpret_cast<bf16*>(args.input_list[tensor_id]);
  const fp32* scale_ptr_1 = reinterpret_cast<fp32*>(args.scale_list_1[tensor_id]);
  const fp32 scale_1 = scale_ptr_1 == nullptr ? 1 : *scale_ptr_1;
  const fp32* scale_ptr_2 = reinterpret_cast<fp32*>(args.scale_list_2[tensor_id]);
  const fp32 scale_2 = scale_ptr_2 == nullptr ? 1 : *scale_ptr_2;
  const int num_elements = args.num_elements_list[tensor_id];

  const int sub_bid = bid - args.block_range[tensor_id];

  using VecT = aligned_vector<bf16, kNumElementsPerThread>;

  bf16 src[kNumElementsPerThread];

  int id = sub_bid * kNumElementsPerBlock + tid * kNumElementsPerThread;
  if constexpr (aligned) {
    VecT* value = reinterpret_cast<VecT*>(&src);
    *value = *reinterpret_cast<const VecT*>(&input[id]);
  } else {
    for (int i = 0; i < kNumElementsPerThread; ++i) {
      int j = id + i;
      if (j < num_elements) {
        src[i] = input[j];
      }
    }
  }
  for (int i = 0; i < kNumElementsPerThread; ++i) {
    src[i] *= __float2bfloat16(scale_1 * scale_2);
  }
  if constexpr (aligned) {
    VecT* value = reinterpret_cast<VecT*>(&src);
    *reinterpret_cast<VecT*>(&input[id]) = *value;
  } else {
    for (int i = 0; i < kNumElementsPerThread; ++i) {
      int j = id + i;
      if (j < num_elements) {
        input[j] = src[i];
      }
    }
  }
}

}  // namespace

void MultiQuantize(std::vector<at::Tensor> input_list, std::vector<at::Tensor> output_list,
                   std::vector<at::Tensor> scale_list, std::vector<at::Tensor> amax_list) {
  if (input_list.empty()) {
    return;
  }

  TORCH_CHECK(input_list.size() <= kMaxTensorsPerKernel);

  // Add tensors to kernel argument struct
  MultiQuantizeArgs kernel_args_aligned, kernel_args_unaligned;
  kernel_args_aligned.num_tensors = 0;
  kernel_args_aligned.block_range[0] = 0;
  kernel_args_unaligned.num_tensors = 0;
  kernel_args_unaligned.block_range[0] = 0;
  constexpr int input_vec_alignment = std::alignment_of<aligned_vector<bf16, kNumElementsPerThread>>::value;
  constexpr int output_vec_alignment = std::alignment_of<aligned_vector<fp8e4m3, kNumElementsPerThread>>::value;
  for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {
    TORCH_CHECK(input_list[tensor_id].scalar_type() == torch::kBFloat16);
    TORCH_CHECK(output_list[tensor_id].scalar_type() == torch::kFloat8_e4m3fn ||
                output_list[tensor_id].scalar_type() == torch::kFloat8_e5m2);
    const int num_elements = input_list[tensor_id].numel();
    const int num_blocks = (num_elements + kNumElementsPerBlock - 1) / kNumElementsPerBlock;
    TORCH_CHECK(num_blocks <= kMaxBlocks);  // Not likely to happen
    const bool aligned = (num_elements % kNumElementsPerThread == 0 &&
                          reinterpret_cast<uint64_t>(input_list[tensor_id].data_ptr()) % input_vec_alignment == 0 &&
                          reinterpret_cast<uint64_t>(output_list[tensor_id].data_ptr()) % output_vec_alignment == 0);
    auto& kernel_args = aligned ? kernel_args_aligned : kernel_args_unaligned;

    // Add tensor to kernel argument struct
    const int pos = kernel_args.num_tensors;
    kernel_args.input_list[pos] = const_cast<void*>(input_list[tensor_id].data_ptr());
    kernel_args.output_list[pos] = output_list[tensor_id].data_ptr();
    kernel_args.scale_list[pos] = scale_list[tensor_id].data_ptr();
    kernel_args.amax_list[pos] = amax_list[tensor_id].data_ptr();
    kernel_args.num_elements_list[pos] = num_elements;
    kernel_args.block_range[pos + 1] = kernel_args.block_range[pos] + num_blocks;
    kernel_args.num_tensors++;
  }

  // Launch kernel
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (kernel_args_aligned.num_tensors > 0) {
    if (output_list[0].scalar_type() == torch::kFloat8_e4m3fn) {
      multi_quantize_kernel<true, fp8e4m3>
          <<<kernel_args_aligned.block_range[kernel_args_aligned.num_tensors], kThreadsPerBlock, 0, stream>>>(
              kernel_args_aligned);
    } else {
      multi_quantize_kernel<true, fp8e5m2>
          <<<kernel_args_aligned.block_range[kernel_args_aligned.num_tensors], kThreadsPerBlock, 0, stream>>>(
              kernel_args_aligned);
    }
  }
  if (kernel_args_unaligned.num_tensors > 0) {
    if (output_list[0].scalar_type() == torch::kFloat8_e4m3fn) {
      multi_quantize_kernel<false, fp8e4m3>
          <<<kernel_args_unaligned.block_range[kernel_args_unaligned.num_tensors], kThreadsPerBlock, 0, stream>>>(
              kernel_args_unaligned);
    } else {
      multi_quantize_kernel<false, fp8e5m2>
          <<<kernel_args_unaligned.block_range[kernel_args_unaligned.num_tensors], kThreadsPerBlock, 0, stream>>>(
              kernel_args_unaligned);
    }
  }
}

void MultiScaleMul(std::vector<at::Tensor> input_list, std::vector<at::Tensor> scale_list_1,
                   std::vector<at::Tensor> scale_list_2) {
  if (input_list.empty()) {
    return;
  }

  TORCH_CHECK(input_list.size() <= kMaxTensorsPerKernel);

  // Add tensors to kernel argument struct
  MultiScaleMulArgs kernel_args_aligned, kernel_args_unaligned;
  kernel_args_aligned.num_tensors = 0;
  kernel_args_aligned.block_range[0] = 0;
  kernel_args_unaligned.num_tensors = 0;
  kernel_args_unaligned.block_range[0] = 0;
  constexpr int vec_alignment = std::alignment_of<aligned_vector<bf16, kNumElementsPerThread>>::value;
  for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {
    TORCH_CHECK(input_list[tensor_id].scalar_type() == torch::kBFloat16);
    const int num_elements = input_list[tensor_id].numel();
    const int num_blocks = (num_elements + kNumElementsPerBlock - 1) / kNumElementsPerBlock;
    TORCH_CHECK(num_blocks <= kMaxBlocks);  // Not likely to happen
    const bool aligned = num_elements % kNumElementsPerThread == 0 &&
                         reinterpret_cast<uint64_t>(input_list[tensor_id].data_ptr()) % vec_alignment == 0;
    auto& kernel_args = aligned ? kernel_args_aligned : kernel_args_unaligned;

    // Add tensor to kernel argument struct
    const int pos = kernel_args.num_tensors;
    kernel_args.input_list[pos] = const_cast<void*>(input_list[tensor_id].data_ptr());
    kernel_args.scale_list_1[pos] = scale_list_1[tensor_id].data_ptr();
    kernel_args.scale_list_2[pos] = scale_list_2[tensor_id].data_ptr();
    kernel_args.num_elements_list[pos] = num_elements;
    kernel_args.block_range[pos + 1] = kernel_args.block_range[pos] + num_blocks;
    kernel_args.num_tensors++;
  }

  // Launch kernel
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  if (kernel_args_aligned.num_tensors > 0) {
    multi_scale_mul_kernel<true>
        <<<kernel_args_aligned.block_range[kernel_args_aligned.num_tensors], kThreadsPerBlock, 0, stream>>>(
            kernel_args_aligned);
  }
  if (kernel_args_unaligned.num_tensors > 0) {
    multi_scale_mul_kernel<false>
        <<<kernel_args_unaligned.block_range[kernel_args_unaligned.num_tensors], kThreadsPerBlock, 0, stream>>>(
            kernel_args_unaligned);
  }
}

}  // namespace fp8_gmm
