#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

namespace fp8_gmm {

using fp32 = float;
using bf16 = nv_bfloat16;
using fp8e4m3 = __nv_fp8_e4m3;
using fp8e5m2 = __nv_fp8_e5m2;

constexpr int THREADS_PER_WARP = 32;

struct uint16 {
  uint4 u;
  uint4 v;
  uint4 s;
  uint4 t;
};

struct uint8 {
  uint4 u;
  uint4 v;
};

template <int BYTES>
struct BytesToType {};

template <>
struct BytesToType<64> {
  using Type = uint16;
  static_assert(sizeof(Type) == 64);
};

template <>
struct BytesToType<32> {
  using Type = uint8;
  static_assert(sizeof(Type) == 32);
};

template <>
struct BytesToType<16> {
  using Type = uint4;
  static_assert(sizeof(Type) == 16);
};

template <>
struct BytesToType<8> {
  using Type = uint64_t;
  static_assert(sizeof(Type) == 8);
};

template <>
struct BytesToType<4> {
  using Type = uint32_t;
  static_assert(sizeof(Type) == 4);
};

template <>
struct BytesToType<2> {
  using Type = uint16_t;
  static_assert(sizeof(Type) == 2);
};

template <>
struct BytesToType<1> {
  using Type = uint8_t;
  static_assert(sizeof(Type) == 1);
};

template <typename Elt_type, uint32_t NUM_ELT>
struct Vec {
  enum { BYTES = NUM_ELT * sizeof(Elt_type) };

  using Vec_type = typename BytesToType<BYTES>::Type;
  using type = Elt_type;

  using Alias_type = union {
    Vec_type vec;
    Elt_type elt[NUM_ELT];
  };

  Alias_type data;

  template <typename S>
  inline __device__ void to(Vec<S, NUM_ELT>& other) {  // NOLINT(*)
#pragma unroll
    for (int it = 0; it < NUM_ELT; it++) {
      other.data.elt[it] = S(this->data.elt[it]);
    }
  }

  template <typename Op>
  inline __device__ void assign(const Op& op) {
#pragma unroll
    for (int it = 0; it < NUM_ELT; it++) {
      this->data.elt[it] = op(it);
    }
  }

  // Pointer is cast to vector type
  inline __device__ void load_from(const void* base_ptr, size_t idx = 0) {
    this->data.vec = static_cast<const Vec_type*>(base_ptr)[idx];
  }

  // Pointer is cast to vector type
  inline __device__ void store_to(void* base_ptr, size_t idx = 0) const {
    static_cast<Vec_type*>(base_ptr)[idx] = this->data.vec;
  }

  // Pointer is cast to element type. Loads min(count, NUM_ELT)
  // elements and any remaining elements are set to zero.
  inline __device__ void load_from_elts(const void* base_ptr, size_t idx = 0, size_t count = NUM_ELT) {
    const Elt_type* elt_ptr = static_cast<const Elt_type*>(base_ptr) + idx;
    if (count < NUM_ELT || reinterpret_cast<uint64_t>(elt_ptr) % BYTES != 0) {
#pragma unroll
      for (int it = 0; it < NUM_ELT; it++) {
        this->data.elt[it] = (it < count ? elt_ptr[it] : Elt_type(0.f));
      }
    } else {
      this->load_from(elt_ptr);
    }
  }

  // Pointer is cast to element type. Stores min(count, NUM_ELT)
  // elements.
  inline __device__ void store_to_elts(void* base_ptr, size_t idx = 0, size_t count = NUM_ELT) const {
    Elt_type* elt_ptr = static_cast<Elt_type*>(base_ptr) + idx;
    if (count < NUM_ELT || reinterpret_cast<uint64_t>(elt_ptr) % BYTES != 0) {
#pragma unroll
      for (int it = 0; it < NUM_ELT; it++) {
        if (it < count) {
          elt_ptr[it] = this->data.elt[it];
        }
      }
    } else {
      this->store_to(elt_ptr);
    }
  }

  inline __device__ void clear() {
#pragma unroll
    for (int it = 0; it < NUM_ELT; it++) {
      this->data.elt[it] = Elt_type(0.f);
    }
  }
};

template <int num_elems>
__device__ __forceinline__ fp32 warp_reduce_max(const fp32 m) {
  fp32 tmp = m;
#pragma unroll
  for (int delta = num_elems / 2; delta > 0; delta /= 2) {
    const fp32 other_m = __shfl_down_sync(0xFFFFFFFF, tmp, delta);
    __builtin_assume(tmp >= 0);
    __builtin_assume(other_m >= 0);
    tmp = fmaxf(tmp, other_m);
  }
  return tmp;
}

template <int num_warps, typename compute_t>
__device__ __forceinline__ compute_t reduce_max(const compute_t m, const int warpid) {
  __shared__ fp32 staging[num_warps];
  constexpr int warp_size = 32;
  const fp32 my_max = m;
  const fp32 my_warp_max = warp_reduce_max<warp_size>(my_max);
  if (threadIdx.x % 32 == 0) {
    staging[warpid] = my_warp_max;
  }
  __syncthreads();
  compute_t result = 0;
  if (warpid == 0) {
    const fp32 my_max = threadIdx.x < num_warps ? staging[threadIdx.x] : 0;
    result = warp_reduce_max<num_warps>(my_max);
  }
  return result;
}

__device__ __forceinline__ void atomicMaxFloat(fp32* addr, const fp32 value) {
  atomicMax(reinterpret_cast<int*>(addr), __float_as_int(value));
}

}  // namespace fp8_gmm
