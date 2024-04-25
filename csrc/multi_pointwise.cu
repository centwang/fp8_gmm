#include "multi_pointwise.h"

#include "utils.cuh"

namespace fp8_gmm {

namespace {

// Parameters to tune
constexpr int kThreadsPerBlock = 256;
constexpr int kNumElementsPerThread = 8;
constexpr int kNumElementsPerBlock = kThreadsPerBlock * kNumElementsPerThread;
constexpr int kMaxTensorsPerKernel = 64;

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

template <bool aligned, typename OType>
__global__ void __launch_bounds__(kThreadsPerBlock) multi_quantize_kernel(MultiQuantizeArgs args) {
  using IVec = Vec<bf16, kNumElementsPerThread>;
  using OVec = Vec<OType, kNumElementsPerThread>;

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

  fp32 max = 0;
  IVec local_input;
  OVec local_output;

  int id = sub_bid * kNumElementsPerBlock + tid * kNumElementsPerThread;
  if constexpr (aligned) {
    local_input.load_from(input + id);
  } else {
    local_input.clear();
#pragma unroll
    for (int i = 0; i < kNumElementsPerThread; ++i) {
      int j = id + i;
      if (j < num_elements) {
        local_input.data.elt[i] = input[j];
      }
    }
  }
#pragma unroll
  for (int i = 0; i < kNumElementsPerThread; ++i) {
    fp32 val = fp32(local_input.data.elt[i]);
    max = fmaxf(fabsf(val), max);
    local_output.data.elt[i] = static_cast<OType>(val * scale);
  }
  if constexpr (aligned) {
    local_output.store_to(output + id);
  } else {
#pragma unroll
    for (int i = 0; i < kNumElementsPerThread; ++i) {
      int j = id + i;
      if (j < num_elements) {
        output[j] = local_output.data.elt[i];
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
  using IVec = Vec<bf16, kNumElementsPerThread>;

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

  IVec local_input;

  int id = sub_bid * kNumElementsPerBlock + tid * kNumElementsPerThread;
  if constexpr (aligned) {
    local_input.load_from(input + id);
  } else {
    local_input.clear();
#pragma unroll
    for (int i = 0; i < kNumElementsPerThread; ++i) {
      int j = id + i;
      if (j < num_elements) {
        local_input.data.elt[i] = input[j];
      }
    }
  }
#pragma unroll
  for (int i = 0; i < kNumElementsPerThread; ++i) {
    local_input.data.elt[i] = bf16(fp32(local_input.data.elt[i]) * scale_1 * scale_2);
  }
  if constexpr (aligned) {
    local_input.store_to(input + id);
  } else {
#pragma unroll
    for (int i = 0; i < kNumElementsPerThread; ++i) {
      int j = id + i;
      if (j < num_elements) {
        input[j] = local_input.data.elt[i];
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
  for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {
    TORCH_CHECK(input_list[tensor_id].scalar_type() == torch::kBFloat16);
    TORCH_CHECK(output_list[tensor_id].scalar_type() == torch::kFloat8_e4m3fn ||
                output_list[tensor_id].scalar_type() == torch::kFloat8_e5m2);
    const int num_elements = input_list[tensor_id].numel();
    const int num_blocks = (num_elements + kNumElementsPerBlock - 1) / kNumElementsPerBlock;
    const bool aligned = num_blocks * kNumElementsPerBlock == num_elements;
    auto& kernel_args = aligned ? kernel_args_aligned : kernel_args_unaligned;

    // Add tensor to kernel argument struct
    const int pos = kernel_args.num_tensors;
    kernel_args.input_list[pos] = input_list[tensor_id].data_ptr();
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
  for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {
    TORCH_CHECK(input_list[tensor_id].scalar_type() == torch::kBFloat16);
    const int num_elements = input_list[tensor_id].numel();
    const int num_blocks = (num_elements + kNumElementsPerBlock - 1) / kNumElementsPerBlock;
    const bool aligned = num_blocks * kNumElementsPerBlock == num_elements;
    auto& kernel_args = aligned ? kernel_args_aligned : kernel_args_unaligned;

    // Add tensor to kernel argument struct
    const int pos = kernel_args.num_tensors;
    kernel_args.input_list[pos] = input_list[tensor_id].data_ptr();
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
