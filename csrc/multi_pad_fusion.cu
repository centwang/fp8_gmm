#include "multi_pad_fusion.h"

#include "utils.cuh"

namespace fp8_gmm {

namespace {

// Parameters to tune
constexpr int kNumWarpsPerTile = 4;
constexpr int kNumThreadsPerBlock = THREADS_PER_WARP * kNumWarpsPerTile;
constexpr int kDesiredLoadSize = 8;
constexpr int kDesiredStoreSize = 8;
constexpr int kMaxTensorsPerKernel = 64;  // Args must be <4 KB

struct MultiPadBaseArgs {
  void* input_list[kMaxTensorsPerKernel];
  void* transpose_list[kMaxTensorsPerKernel];
  int num_rows_list[kMaxTensorsPerKernel];
  int row_length_list[kMaxTensorsPerKernel];
  int padded_rows_list[kMaxTensorsPerKernel];
  int block_range[kMaxTensorsPerKernel + 1];
  int num_tensors;
};

struct MultiPadCastTransposeArgs : MultiPadBaseArgs {
  void* output_list[kMaxTensorsPerKernel];
  void* cast_list[kMaxTensorsPerKernel];
  void* scale_list[kMaxTensorsPerKernel];
  void* amax_list[kMaxTensorsPerKernel];
};

struct MultiPadCastTransposeDgeluArgs : MultiPadCastTransposeArgs {
  void* gelu_input_list[kMaxTensorsPerKernel];
};

// Approximation compute.
template <typename IType>
__device__ inline fp32 dgelu(const IType val) {
  const fp32 cval = val;
  const fp32 tanh_out = tanhf(0.79788456f * cval * (1.f + 0.044715f * cval * cval));
  return 0.5f * cval * ((1.f - tanh_out * tanh_out) * (0.79788456f + 0.1070322243f * cval * cval)) +
         0.5f * (1.f + tanh_out);
}

template <int nvec_in, int nvec_out, bool aligned, typename Type>
__global__ void __launch_bounds__(kNumThreadsPerBlock) multi_pad_transpose_kernel(MultiPadBaseArgs args) {
  using IVec = Vec<Type, nvec_in>;
  using OVec = Vec<Type, nvec_out>;

  // Thread indices
  // Note: Block is interpreted as a warp_size x num_warps grid
  constexpr int bdimx = THREADS_PER_WARP;
  constexpr int bdimy = kNumWarpsPerTile;
  const int tid = threadIdx.x;
  const int tidx = tid % bdimx;
  const int tidy = tid / bdimx;
  const int bid = blockIdx.x;

  // Input tensors are divided into tiles
  // Note: Each tile is a warp_size x warp_size grid of nvec_out x nvec_in subtiles
  constexpr int tile_dim_m = THREADS_PER_WARP * nvec_out;
  constexpr int tile_dim_n = THREADS_PER_WARP * nvec_in;

  // Number of nvec_out x nvec_in subtiles for each thread to
  // load/store
  constexpr int num_iterations = THREADS_PER_WARP / kNumWarpsPerTile;

  // Find tensor corresponding to block
  int tensor_id = 0;
  while (args.block_range[tensor_id + 1] <= bid) {
    ++tensor_id;
  }
  const Type* input = reinterpret_cast<const Type*>(args.input_list[tensor_id]);
  Type* transpose = reinterpret_cast<Type*>(args.transpose_list[tensor_id]);
  const int num_rows = args.num_rows_list[tensor_id];
  const int row_length = args.row_length_list[tensor_id];
  const int padded_rows = args.padded_rows_list[tensor_id];

  // Position of tile within tensor
  const int num_tiles_m = (num_rows + tile_dim_m - 1) / tile_dim_m;
  const int tile_id = bid - args.block_range[tensor_id];
  const int tile_id_m = tile_id % num_tiles_m;
  const int tile_id_n = tile_id / num_tiles_m;
  const int tile_row = tile_id_m * tile_dim_m;
  const int tile_col = tile_id_n * tile_dim_n;

  // Load input and store to registers
  // Note: Each thread loads num_iterations subtiles and transposes in
  // registers.
  OVec local_transpose[nvec_in][num_iterations];
#pragma unroll
  for (int iter = 0; iter < num_iterations; ++iter) {
    const int i1 = tidy + iter * bdimy;
    const int j1 = tidx;
#pragma unroll
    for (int i2 = 0; i2 < nvec_out; ++i2) {
      const int row = tile_row + i1 * nvec_out + i2;
      const int col = tile_col + j1 * nvec_in;
      IVec local_input;
      if constexpr (aligned) {
        local_input.load_from(&input[row * row_length + col]);
      } else {
        local_input.clear();
        if (row < num_rows) {
#pragma unroll
          for (int j2 = 0; j2 < nvec_in; ++j2) {
            if (col + j2 < row_length) {
              local_input.data.elt[j2] = input[row * row_length + col + j2];
            }
          }
        }
      }
#pragma unroll
      for (int j2 = 0; j2 < nvec_in; ++j2) {
        local_transpose[j2][iter].data.elt[i2] = local_input.data.elt[j2];
      }
    }
  }

  // Copy transposed output from registers to global memory
  __shared__ OVec shared_transpose[THREADS_PER_WARP][THREADS_PER_WARP + 1];
#pragma unroll
  for (int j2 = 0; j2 < nvec_in; ++j2) {
#pragma unroll
    for (int iter = 0; iter < num_iterations; ++iter) {
      const int i1 = tidy + iter * bdimy;
      const int j1 = tidx;
      shared_transpose[j1][i1] = local_transpose[j2][iter];
    }
    __syncthreads();
#pragma unroll
    for (int iter = 0; iter < num_iterations; ++iter) {
      const int i1 = tidx;
      const int j1 = tidy + iter * bdimy;
      const int row = tile_row + i1 * nvec_out;
      const int col = tile_col + j1 * nvec_in + j2;
      if constexpr (aligned) {
        shared_transpose[j1][i1].store_to(&transpose[col * padded_rows + row]);
      } else {
        if (col < row_length) {
#pragma unroll
          for (int i2 = 0; i2 < nvec_out; ++i2) {
            if (row + i2 < padded_rows) {
              transpose[col * padded_rows + row + i2] = shared_transpose[j1][i1].data.elt[i2];
            }
          }
        }
      }
    }
    __syncthreads();
  }
}

template <int nvec_in, int nvec_out, bool aligned, typename OType>
__global__ void __launch_bounds__(kNumThreadsPerBlock) multi_pad_cast_transpose_kernel(MultiPadCastTransposeArgs args) {
  using IVec = Vec<bf16, nvec_in>;
  using OVecC = Vec<OType, nvec_in>;
  using OVecT = Vec<OType, nvec_out>;

  // Thread indices
  // Note: Block is interpreted as a warp_size x num_warps grid
  constexpr int bdimx = THREADS_PER_WARP;
  constexpr int bdimy = kNumWarpsPerTile;
  const int tid = threadIdx.x;
  const int tidx = tid % bdimx;
  const int tidy = tid / bdimx;
  const int bid = blockIdx.x;

  // Input tensors are divided into tiles
  // Note: Each tile is a warp_size x warp_size grid of nvec_out x nvec_in subtiles
  constexpr int tile_dim_m = THREADS_PER_WARP * nvec_out;
  constexpr int tile_dim_n = THREADS_PER_WARP * nvec_in;

  // Number of nvec_out x nvec_in subtiles for each thread to
  // load/store
  constexpr int n_iterations = THREADS_PER_WARP / kNumWarpsPerTile;

  // Find tensor corresponding to block
  int tensor_id = 0;
  while (args.block_range[tensor_id + 1] <= bid) {
    ++tensor_id;
  }
  const bf16* input = reinterpret_cast<const bf16*>(args.input_list[tensor_id]);
  OType* output = reinterpret_cast<OType*>(args.output_list[tensor_id]);
  OType* transpose = reinterpret_cast<OType*>(args.transpose_list[tensor_id]);
  const fp32* scale_ptr = reinterpret_cast<fp32*>(args.scale_list[tensor_id]);
  const fp32 scale = scale_ptr == nullptr ? 1 : *scale_ptr;
  fp32* amax = reinterpret_cast<fp32*>(args.amax_list[tensor_id]);
  const int num_rows = args.num_rows_list[tensor_id];
  const int row_length = args.row_length_list[tensor_id];
  const int padded_rows = args.padded_rows_list[tensor_id];

  // Find position of tile within tensor
  const int num_tiles_n = (row_length + tile_dim_n - 1) / tile_dim_n;
  const int tile_id = bid - args.block_range[tensor_id];
  const int tile_id_m = tile_id / num_tiles_n;
  const int tile_id_n = tile_id % num_tiles_n;
  const int tile_row = tile_id_m * tile_dim_m;
  const int tile_col = tile_id_n * tile_dim_n;

  // Load input and store to registers
  // Note: Each thread loads n_iterations subtiles, casts to output
  // type, and transposes in registers.
  OVecT local_transpose[nvec_in][n_iterations];
  fp32 local_amax = 0;
#pragma unroll
  for (int iter = 0; iter < n_iterations; ++iter) {
    const int i1 = tidy + iter * bdimy;
    const int j1 = tidx;
#pragma unroll
    for (int i2 = 0; i2 < nvec_out; ++i2) {
      const int row = tile_row + i1 * nvec_out + i2;
      const int col = tile_col + j1 * nvec_in;
      IVec local_input;
      OVecC local_output;
      if constexpr (aligned) {
        local_input.load_from(&input[row * row_length + col]);
      } else {
        local_input.clear();
        if (row < num_rows) {
#pragma unroll
          for (int j2 = 0; j2 < nvec_in; ++j2) {
            if (col + j2 < row_length) {
              local_input.data.elt[j2] = input[row * row_length + col + j2];
            }
          }
        }
      }
#pragma unroll
      for (int j2 = 0; j2 < nvec_in; ++j2) {
        const fp32 x = fp32(local_input.data.elt[j2]);
        const OType y = OType(scale * x);
        local_output.data.elt[j2] = y;
        local_transpose[j2][iter].data.elt[i2] = y;
        local_amax = fmaxf(fabsf(x), local_amax);
      }
      if constexpr (aligned) {
        local_output.store_to(&output[row * row_length + col]);
      } else {
        if (row < num_rows) {
#pragma unroll
          for (int j2 = 0; j2 < nvec_in; ++j2) {
            if (col + j2 < row_length) {
              output[row * row_length + col + j2] = local_output.data.elt[j2];
            }
          }
        }
      }
    }
  }

  // Copy transposed output from registers to global memory
  __shared__ OVecT shared_transpose[THREADS_PER_WARP][THREADS_PER_WARP + 1];
#pragma unroll
  for (int j2 = 0; j2 < nvec_in; ++j2) {
#pragma unroll
    for (int iter = 0; iter < n_iterations; ++iter) {
      const int i1 = tidy + iter * bdimy;
      const int j1 = tidx;
      shared_transpose[j1][i1] = local_transpose[j2][iter];
    }
    __syncthreads();
#pragma unroll
    for (int iter = 0; iter < n_iterations; ++iter) {
      const int i1 = tidx;
      const int j1 = tidy + iter * bdimy;
      const int row = tile_row + i1 * nvec_out;
      const int col = tile_col + j1 * nvec_in + j2;
      if constexpr (aligned) {
        shared_transpose[j1][i1].store_to(&transpose[col * padded_rows + row]);
      } else {
        if (col < row_length) {
#pragma unroll
          for (int i2 = 0; i2 < nvec_out; ++i2) {
            if (row + i2 < padded_rows) {
              transpose[col * padded_rows + row + i2] = shared_transpose[j1][i1].data.elt[i2];
            }
          }
        }
      }
    }
    __syncthreads();
  }

  // Finalize fp8 factors
  local_amax = reduce_max<kNumWarpsPerTile>(local_amax, tidy);
  if (tid == 0) {
    static_assert(std::is_same<fp32, float>::value);
    if (amax != nullptr) atomicMaxFloat(amax, local_amax);
  }
}

// Assume gelu_input is BFloat16 type.
template <int nvec_in, int nvec_out, bool aligned, typename OType>
__global__ void __launch_bounds__(kNumThreadsPerBlock)
    multi_pad_cast_transpose_dgelu_kernel(MultiPadCastTransposeDgeluArgs args) {
  using IVec = Vec<bf16, nvec_in>;
  using OVecC = Vec<OType, nvec_in>;
  using OVecT = Vec<OType, nvec_out>;

  // Thread indices
  // Note: Block is interpreted as a warp_size x num_warps grid
  constexpr int bdimx = THREADS_PER_WARP;
  constexpr int bdimy = kNumWarpsPerTile;
  const int tid = threadIdx.x;
  const int tidx = tid % bdimx;
  const int tidy = tid / bdimx;
  const int bid = blockIdx.x;

  // Input tensors are divided into tiles
  // Note: Each tile is a warp_size x warp_size grid of nvec_out x nvec_in subtiles
  constexpr int tile_dim_m = THREADS_PER_WARP * nvec_out;
  constexpr int tile_dim_n = THREADS_PER_WARP * nvec_in;

  // Number of nvec_out x nvec_in subtiles for each thread to
  // load/store
  constexpr int n_iterations = THREADS_PER_WARP / kNumWarpsPerTile;

  // Find tensor corresponding to block
  int tensor_id = 0;
  while (args.block_range[tensor_id + 1] <= bid) {
    ++tensor_id;
  }
  const bf16* input = reinterpret_cast<const bf16*>(args.input_list[tensor_id]);
  const bf16* gelu_input = reinterpret_cast<const bf16*>(args.gelu_input_list[tensor_id]);
  OType* output = reinterpret_cast<OType*>(args.output_list[tensor_id]);
  OType* transpose = reinterpret_cast<OType*>(args.transpose_list[tensor_id]);
  const fp32* scale_ptr = reinterpret_cast<fp32*>(args.scale_list[tensor_id]);
  const fp32 scale = scale_ptr == nullptr ? 1 : *scale_ptr;
  fp32* amax = reinterpret_cast<fp32*>(args.amax_list[tensor_id]);
  const int num_rows = args.num_rows_list[tensor_id];
  const int row_length = args.row_length_list[tensor_id];
  const int padded_rows = args.padded_rows_list[tensor_id];

  // Find position of tile within tensor
  const int num_tiles_n = (row_length + tile_dim_n - 1) / tile_dim_n;
  const int tile_id = bid - args.block_range[tensor_id];
  const int tile_id_m = tile_id / num_tiles_n;
  const int tile_id_n = tile_id % num_tiles_n;
  const int tile_row = tile_id_m * tile_dim_m;
  const int tile_col = tile_id_n * tile_dim_n;

  // Load input and store to registers
  // Note: Each thread loads n_iterations subtiles, casts to output
  // type, and transposes in registers.
  OVecT local_transpose[nvec_in][n_iterations];
  fp32 local_amax = 0;
#pragma unroll
  for (int iter = 0; iter < n_iterations; ++iter) {
    const int i1 = tidy + iter * bdimy;
    const int j1 = tidx;
#pragma unroll
    for (int i2 = 0; i2 < nvec_out; ++i2) {
      const int row = tile_row + i1 * nvec_out + i2;
      const int col = tile_col + j1 * nvec_in;
      IVec local_input;
      IVec local_gelu_input;
      OVecC local_output;
      if constexpr (aligned) {
        local_input.load_from(&input[row * row_length + col]);
        local_gelu_input.load_from(&gelu_input[row * row_length + col]);
      } else {
        local_input.clear();
        local_gelu_input.clear();
        if (row < num_rows) {
#pragma unroll
          for (int j2 = 0; j2 < nvec_in; ++j2) {
            if (col + j2 < row_length) {
              local_input.data.elt[j2] = input[row * row_length + col + j2];
              local_gelu_input.data.elt[j2] = gelu_input[row * row_length + col + j2];
            }
          }
        }
      }
#pragma unroll
      for (int j2 = 0; j2 < nvec_in; ++j2) {
        const fp32 x = fp32(local_input.data.elt[j2]) * dgelu(local_gelu_input.data.elt[j2]);
        const OType y = OType(scale * x);
        local_output.data.elt[j2] = y;
        local_transpose[j2][iter].data.elt[i2] = y;
        local_amax = fmaxf(fabsf(x), local_amax);
      }
      if constexpr (aligned) {
        local_output.store_to(&output[row * row_length + col]);
      } else {
        if (row < num_rows) {
#pragma unroll
          for (int j2 = 0; j2 < nvec_in; ++j2) {
            if (col + j2 < row_length) {
              output[row * row_length + col + j2] = local_output.data.elt[j2];
            }
          }
        }
      }
    }
  }

  // Copy transposed output from registers to global memory
  __shared__ OVecT shared_transpose[THREADS_PER_WARP][THREADS_PER_WARP + 1];
#pragma unroll
  for (int j2 = 0; j2 < nvec_in; ++j2) {
#pragma unroll
    for (int iter = 0; iter < n_iterations; ++iter) {
      const int i1 = tidy + iter * bdimy;
      const int j1 = tidx;
      shared_transpose[j1][i1] = local_transpose[j2][iter];
    }
    __syncthreads();
#pragma unroll
    for (int iter = 0; iter < n_iterations; ++iter) {
      const int i1 = tidx;
      const int j1 = tidy + iter * bdimy;
      const int row = tile_row + i1 * nvec_out;
      const int col = tile_col + j1 * nvec_in + j2;
      if constexpr (aligned) {
        shared_transpose[j1][i1].store_to(&transpose[col * padded_rows + row]);
      } else {
        if (col < row_length) {
#pragma unroll
          for (int i2 = 0; i2 < nvec_out; ++i2) {
            if (row + i2 < padded_rows) {
              transpose[col * padded_rows + row + i2] = shared_transpose[j1][i1].data.elt[i2];
            }
          }
        }
      }
    }
    __syncthreads();
  }

  // Finalize fp8 factors
  local_amax = reduce_max<kNumWarpsPerTile>(local_amax, tidy);
  if (tid == 0) {
    static_assert(std::is_same<fp32, float>::value);
    if (amax != nullptr) atomicMaxFloat(amax, local_amax);
  }
}

}  // namespace

void MultiPadTranspose(std::vector<at::Tensor> input_list, std::vector<at::Tensor> transpose_list) {
  if (input_list.empty()) {
    return;
  }

  TORCH_CHECK(input_list.size() <= kMaxTensorsPerKernel);

  // Input matrices are divided into tiles
  // Note: Each tile is a warp_size x warp_size grid of nvec_out x nvec_in subtiles
  const int tile_dim_m = THREADS_PER_WARP * kDesiredStoreSize / sizeof(fp8e4m3);
  const int tile_dim_n = THREADS_PER_WARP * kDesiredLoadSize / sizeof(fp8e4m3);

  // Add tensors to kernel argument struct
  MultiPadBaseArgs kernel_args_aligned, kernel_args_unaligned;
  kernel_args_aligned.num_tensors = 0;
  kernel_args_aligned.block_range[0] = 0;
  kernel_args_unaligned.num_tensors = 0;
  kernel_args_unaligned.block_range[0] = 0;
  for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {
    TORCH_CHECK(input_list[tensor_id].scalar_type() == transpose_list[tensor_id].scalar_type());
    TORCH_CHECK(input_list[tensor_id].scalar_type() == torch::kFloat8_e4m3fn ||
                input_list[tensor_id].scalar_type() == torch::kFloat8_e5m2);
    const int num_rows = input_list[tensor_id].size(0);
    const int row_length = input_list[tensor_id].size(1);
    const int trans_rows = transpose_list[tensor_id].size(0);
    const int trans_cols = transpose_list[tensor_id].size(1);
    TORCH_CHECK(row_length == trans_rows && trans_cols >= num_rows);
    const int num_tiles_m = (trans_cols + tile_dim_m - 1) / tile_dim_m;
    const int num_tiles_n = (row_length + tile_dim_n - 1) / tile_dim_n;
    const int num_tiles = num_tiles_m * num_tiles_n;

    // Figure out whether to use aligned or unaligned kernel
    const bool aligned = ((trans_cols == num_rows) && (num_tiles_m * tile_dim_m == trans_cols) &&
                          (num_tiles_n * tile_dim_n == row_length));
    auto& kernel_args = aligned ? kernel_args_aligned : kernel_args_unaligned;

    // Add tensor to kernel argument struct
    const int pos = kernel_args.num_tensors;
    kernel_args.input_list[pos] = input_list[tensor_id].data_ptr();
    kernel_args.transpose_list[pos] = transpose_list[tensor_id].data_ptr();
    kernel_args.num_rows_list[pos] = num_rows;
    kernel_args.row_length_list[pos] = row_length;
    kernel_args.padded_rows_list[pos] = trans_cols;
    kernel_args.block_range[pos + 1] = kernel_args.block_range[pos] + num_tiles;
    kernel_args.num_tensors++;
  }

  // Launch kernel, since no data compute, always use fp8e4m3 dtype.
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int nvec_in = kDesiredLoadSize / sizeof(fp8e4m3);
  constexpr int nvec_out = kDesiredStoreSize / sizeof(fp8e4m3);
  if (kernel_args_aligned.num_tensors > 0) {
    multi_pad_transpose_kernel<nvec_in, nvec_out, true, fp8e4m3>
        <<<kernel_args_aligned.block_range[kernel_args_aligned.num_tensors], kNumThreadsPerBlock, 0, stream>>>(
            kernel_args_aligned);
  }
  if (kernel_args_unaligned.num_tensors > 0) {
    multi_pad_transpose_kernel<nvec_in, nvec_out, false, fp8e4m3>
        <<<kernel_args_unaligned.block_range[kernel_args_unaligned.num_tensors], kNumThreadsPerBlock, 0, stream>>>(
            kernel_args_unaligned);
  }
}

void MultiPadCastTranspose(std::vector<at::Tensor> input_list, std::vector<at::Tensor> output_list,
                           std::vector<at::Tensor> transpose_list, std::vector<at::Tensor> scale_list,
                           std::vector<at::Tensor> amax_list) {
  if (input_list.empty()) {
    return;
  }

  TORCH_CHECK(input_list.size() <= kMaxTensorsPerKernel);

  // Input matrices are divided into tiles
  // Note: Each tile is a warp_size x warp_size grid of nvec_out x nvec_in subtiles
  const int tile_dim_m = THREADS_PER_WARP * kDesiredStoreSize / sizeof(fp8e4m3);
  const int tile_dim_n = THREADS_PER_WARP * kDesiredLoadSize / sizeof(bf16);

  // Add tensors to kernel argument struct
  MultiPadCastTransposeArgs kernel_args_aligned, kernel_args_unaligned;
  kernel_args_aligned.num_tensors = 0;
  kernel_args_aligned.block_range[0] = 0;
  kernel_args_unaligned.num_tensors = 0;
  kernel_args_unaligned.block_range[0] = 0;
  for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {
    TORCH_CHECK(input_list[tensor_id].scalar_type() == torch::kBFloat16);
    TORCH_CHECK(output_list[tensor_id].scalar_type() == torch::kFloat8_e4m3fn ||
                output_list[tensor_id].scalar_type() == torch::kFloat8_e5m2);
    TORCH_CHECK(output_list[tensor_id].scalar_type() == transpose_list[tensor_id].scalar_type());
    TORCH_CHECK(input_list[tensor_id].sizes() == output_list[tensor_id].sizes());
    const int num_rows = input_list[tensor_id].size(0);
    const int row_length = input_list[tensor_id].size(1);
    const int trans_rows = transpose_list[tensor_id].size(0);
    const int trans_cols = transpose_list[tensor_id].size(1);
    TORCH_CHECK(row_length == trans_rows && trans_cols >= num_rows);
    const int num_tiles_m = (trans_cols + tile_dim_m - 1) / tile_dim_m;
    const int num_tiles_n = (row_length + tile_dim_n - 1) / tile_dim_n;
    const int num_tiles = num_tiles_m * num_tiles_n;

    // Figure out whether to use aligned or unaligned kernel
    const bool aligned = ((trans_cols == num_rows) && (num_tiles_m * tile_dim_m == trans_cols) &&
                          (num_tiles_n * tile_dim_n == row_length));
    auto& kernel_args = aligned ? kernel_args_aligned : kernel_args_unaligned;

    // Add tensor to kernel argument struct
    const int pos = kernel_args.num_tensors;
    kernel_args.input_list[pos] = input_list[tensor_id].data_ptr();
    kernel_args.output_list[pos] = output_list[tensor_id].data_ptr();
    kernel_args.transpose_list[pos] = transpose_list[tensor_id].data_ptr();
    kernel_args.scale_list[pos] = scale_list[tensor_id].data_ptr();
    kernel_args.amax_list[pos] = amax_list[tensor_id].data_ptr();
    kernel_args.num_rows_list[pos] = num_rows;
    kernel_args.row_length_list[pos] = row_length;
    kernel_args.padded_rows_list[pos] = trans_cols;
    kernel_args.block_range[pos + 1] = kernel_args.block_range[pos] + num_tiles;
    kernel_args.num_tensors++;
  }

  // Launch kernel
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int nvec_in = kDesiredLoadSize / sizeof(bf16);
  constexpr int nvec_out = kDesiredStoreSize / sizeof(fp8e4m3);
  if (kernel_args_aligned.num_tensors > 0) {
    if (output_list[0].scalar_type() == torch::kFloat8_e4m3fn) {
      multi_pad_cast_transpose_kernel<nvec_in, nvec_out, true, fp8e4m3>
          <<<kernel_args_aligned.block_range[kernel_args_aligned.num_tensors], kNumThreadsPerBlock, 0, stream>>>(
              kernel_args_aligned);
    } else {
      multi_pad_cast_transpose_kernel<nvec_in, nvec_out, true, fp8e5m2>
          <<<kernel_args_aligned.block_range[kernel_args_aligned.num_tensors], kNumThreadsPerBlock, 0, stream>>>(
              kernel_args_aligned);
    }
  }
  if (kernel_args_unaligned.num_tensors > 0) {
    if (output_list[0].scalar_type() == torch::kFloat8_e4m3fn) {
      multi_pad_cast_transpose_kernel<nvec_in, nvec_out, false, fp8e4m3>
          <<<kernel_args_unaligned.block_range[kernel_args_unaligned.num_tensors], kNumThreadsPerBlock, 0, stream>>>(
              kernel_args_unaligned);
    } else {
      multi_pad_cast_transpose_kernel<nvec_in, nvec_out, false, fp8e5m2>
          <<<kernel_args_unaligned.block_range[kernel_args_unaligned.num_tensors], kNumThreadsPerBlock, 0, stream>>>(
              kernel_args_unaligned);
    }
  }
}

void MultiPadCastTransposeDgelu(std::vector<at::Tensor> input_list, std::vector<at::Tensor> gelu_input_list,
                                std::vector<at::Tensor> output_list, std::vector<at::Tensor> transpose_list,
                                std::vector<at::Tensor> scale_list, std::vector<at::Tensor> amax_list) {
  if (input_list.empty()) {
    return;
  }

  TORCH_CHECK(input_list.size() <= kMaxTensorsPerKernel);

  // Input matrices are divided into tiles
  // Note: Each tile is a warp_size x warp_size grid of nvec_out x nvec_in subtiles
  const int tile_dim_m = THREADS_PER_WARP * kDesiredStoreSize / sizeof(fp8e4m3);
  const int tile_dim_n = THREADS_PER_WARP * kDesiredLoadSize / sizeof(bf16);

  // Add tensors to kernel argument struct
  MultiPadCastTransposeDgeluArgs kernel_args_aligned, kernel_args_unaligned;
  kernel_args_aligned.num_tensors = 0;
  kernel_args_aligned.block_range[0] = 0;
  kernel_args_unaligned.num_tensors = 0;
  kernel_args_unaligned.block_range[0] = 0;
  for (size_t tensor_id = 0; tensor_id < input_list.size(); ++tensor_id) {
    TORCH_CHECK(input_list[tensor_id].scalar_type() == torch::kBFloat16 &&
                gelu_input_list[tensor_id].scalar_type() == torch::kBFloat16);
    TORCH_CHECK(output_list[tensor_id].scalar_type() == torch::kFloat8_e4m3fn ||
                output_list[tensor_id].scalar_type() == torch::kFloat8_e5m2);
    TORCH_CHECK(output_list[tensor_id].scalar_type() == transpose_list[tensor_id].scalar_type());
    TORCH_CHECK(input_list[tensor_id].sizes() == gelu_input_list[tensor_id].sizes() &&
                input_list[tensor_id].sizes() == output_list[tensor_id].sizes());
    const int num_rows = input_list[tensor_id].size(0);
    const int row_length = input_list[tensor_id].size(1);
    const int trans_rows = transpose_list[tensor_id].size(0);
    const int trans_cols = transpose_list[tensor_id].size(1);
    TORCH_CHECK(row_length == trans_rows && trans_cols >= num_rows);
    const int num_tiles_m = (trans_cols + tile_dim_m - 1) / tile_dim_m;
    const int num_tiles_n = (row_length + tile_dim_n - 1) / tile_dim_n;
    const int num_tiles = num_tiles_m * num_tiles_n;

    // Figure out whether to use aligned or unaligned kernel
    const bool aligned = ((trans_cols == num_rows) && (num_tiles_m * tile_dim_m == trans_cols) &&
                          (num_tiles_n * tile_dim_n == row_length));
    auto& kernel_args = aligned ? kernel_args_aligned : kernel_args_unaligned;

    // Add tensor to kernel argument struct
    const int pos = kernel_args.num_tensors;
    kernel_args.input_list[pos] = input_list[tensor_id].data_ptr();
    kernel_args.gelu_input_list[pos] = gelu_input_list[tensor_id].data_ptr();
    kernel_args.output_list[pos] = output_list[tensor_id].data_ptr();
    kernel_args.transpose_list[pos] = transpose_list[tensor_id].data_ptr();
    kernel_args.scale_list[pos] = scale_list[tensor_id].data_ptr();
    kernel_args.amax_list[pos] = amax_list[tensor_id].data_ptr();
    kernel_args.num_rows_list[pos] = num_rows;
    kernel_args.row_length_list[pos] = row_length;
    kernel_args.padded_rows_list[pos] = trans_cols;
    kernel_args.block_range[pos + 1] = kernel_args.block_range[pos] + num_tiles;
    kernel_args.num_tensors++;
  }

  // Launch kernel
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  constexpr int nvec_in = kDesiredLoadSize / sizeof(bf16);
  constexpr int nvec_out = kDesiredStoreSize / sizeof(fp8e4m3);
  if (kernel_args_aligned.num_tensors > 0) {
    if (output_list[0].scalar_type() == torch::kFloat8_e4m3fn) {
      multi_pad_cast_transpose_dgelu_kernel<nvec_in, nvec_out, true, fp8e4m3>
          <<<kernel_args_aligned.block_range[kernel_args_aligned.num_tensors], kNumThreadsPerBlock, 0, stream>>>(
              kernel_args_aligned);
    } else {
      multi_pad_cast_transpose_dgelu_kernel<nvec_in, nvec_out, true, fp8e5m2>
          <<<kernel_args_aligned.block_range[kernel_args_aligned.num_tensors], kNumThreadsPerBlock, 0, stream>>>(
              kernel_args_aligned);
    }
  }
  if (kernel_args_unaligned.num_tensors > 0) {
    if (output_list[0].scalar_type() == torch::kFloat8_e4m3fn) {
      multi_pad_cast_transpose_dgelu_kernel<nvec_in, nvec_out, false, fp8e4m3>
          <<<kernel_args_unaligned.block_range[kernel_args_unaligned.num_tensors], kNumThreadsPerBlock, 0, stream>>>(
              kernel_args_unaligned);
    } else {
      multi_pad_cast_transpose_dgelu_kernel<nvec_in, nvec_out, false, fp8e5m2>
          <<<kernel_args_unaligned.block_range[kernel_args_unaligned.num_tensors], kNumThreadsPerBlock, 0, stream>>>(
              kernel_args_unaligned);
    }
  }
}

}  // namespace fp8_gmm
