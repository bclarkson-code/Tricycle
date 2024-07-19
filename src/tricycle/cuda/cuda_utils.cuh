// Utilities for use in __device__ code

#ifndef CUDA_UTILS_CUH
#define CUDA_UTILS_CUH

#include "cuda_common.h"

// ----------------------------------------------------------------------------
// Packed128 data structure that forces the compiler to use 128-bit loads/stores
// in GPUs that support (the LDG.128 and STS.128 instructions)
// This is a bit similar to the use of float4 in the case of 32-bit floats, but
// supports arbitrary precision.

template <class ElementType> struct alignas(16) Packed128 {
  Packed128() = default;
  __device__ explicit Packed128(int4 bits) {
    static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
    memcpy(&payload, &bits, sizeof(bits));
  }

  __device__ static Packed128 constant(ElementType value) {
    Packed128 result;
    for (int k = 0; k < size; ++k) {
      result.payload[k] = value;
    }
    return result;
  }
  __device__ static Packed128 zeros() { return constant(0.f); }
  __device__ static Packed128 ones() { return constant(1.f); }

  __device__ ElementType &operator[](int index) { return payload[index]; }
  __device__ const ElementType &operator[](int index) const {
    return payload[index];
  }
  __device__ int4 get_bits() const {
    int4 bits;
    static_assert(sizeof(bits) == sizeof(payload), "Size mismatch.");
    memcpy(&bits, &payload, sizeof(bits));
    return bits;
  }
  static constexpr const size_t size = sizeof(int4) / sizeof(ElementType);
  ElementType payload[size];
};

// load a Packed128 from an aligned memory address
template <class ElementType>
__device__ Packed128<ElementType> load128(const ElementType *address) {
  return Packed128<ElementType>{*reinterpret_cast<const int4 *>(address)};
}
// load a Packed128 from an aligned memory address with streaming cache hint
template <class ElementType>
__device__ Packed128<ElementType> load128cs(const ElementType *address) {
  return Packed128<ElementType>{
      __ldcs(reinterpret_cast<const int4 *>(address))};
}
// store a Packed128 to an aligned memory address
template <class ElementType>
__device__ void store128(ElementType *target, Packed128<ElementType> value) {
  *reinterpret_cast<int4 *>(target) = value.get_bits();
}
// store a Packed128 to an aligned memory address with streaming cache hint
template <class ElementType>
__device__ void store128cs(ElementType *target, Packed128<ElementType> value) {
  __stcs(reinterpret_cast<int4 *>(target), value.get_bits());
}
// store a Packed128 to an aligned memory address while caching in L2 but
// bypassing L1
template <class ElementType>
__device__ void store128cg(ElementType *target, Packed128<ElementType> value) {
  __stcg(reinterpret_cast<int4 *>(target), value.get_bits());
}

// short-form typedefs
typedef Packed128<float> f128;
typedef Packed128<floatX> x128;

// ----------------------------------------------------------------------------
// Copy, cast functions

// device functions and the kernel to cast data between types
template <typename Td, typename Ts> __device__ Td cast_value(Ts val);

template <> __device__ float cast_value<float, float>(float val) { return val; }

template <> __device__ float cast_value<float, half>(half val) {
  return __half2float(val);
}

template <>
__device__ float cast_value<float, __nv_bfloat16>(__nv_bfloat16 val) {
  return __bfloat162float(val);
}

template <typename Td, typename Ts>
__global__ void copy_and_cast_kernel(Td *dst, const Ts *src, size_t n,
                                     ptrdiff_t stride_dst,
                                     ptrdiff_t stride_src) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // need to try grid stride looping for more perf later
  if (idx < n) {
    dst[idx + stride_dst * blockIdx.y] =
        cast_value<Td, Ts>(src[idx + stride_src * blockIdx.y]);
  }
}

// //
// ----------------------------------------------------------------------------
// // Warp/Block communication primitives
//
// // warp-level reduction for summing values
// __device__ inline float warpReduceSum(float val) {
//   for (int offset = 16; offset > 0; offset /= 2) {
//     val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
//   }
//   return val;
// }
// // warp-level reduction for finding the maximum value
// __device__ inline float warpReduceMax(float val) {
//   for (int offset = 16; offset > 0; offset /= 2) {
//     val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
//   }
//   return val;
// }
// // requires all 32 threads in the warp to be active, but should work for any
// // block size uses non-dynamic shared memory so every call increases shared
// // memory requirements by 128 bytes the fact it's unique shared memory allows
// us
// // to avoid an extra __syncthreads() call at the end but if called inside a
// // loop, the shared memory will be implicitly reused, so set final_sync to 1
// using reduction_func_t = float (*)(float);
// template <reduction_func_t warp_reduction>
// __device__ inline float blockReduce(float val, bool final_sync = false,
//                                     float out_of_bounds = 0.0f) {
//   // two reductions of up to 1024 threads:
//   // 1) inside warp (shuffle), 2) cross-warp (shared memory), 3) inside warp
//   // (shuffle)
//   __shared__ float shared_val[WARP_SIZE];
//   const int lane_id = threadIdx.x % WARP_SIZE;
//   const int warp_id = threadIdx.x / WARP_SIZE;
//   const int num_warps = blockDim.x / WARP_SIZE;
//
//   float warp_val = warp_reduction(val);
//   if (lane_id == 0) {
//     shared_val[warp_id] = warp_val;
//   }
//   __syncthreads();
//   warp_val = (lane_id < num_warps) ? shared_val[lane_id] : out_of_bounds;
//   float block_val = warp_reduction(warp_val);
//
//   if (final_sync) {
//     __syncthreads(); // only needed in loops when effectively reusing shared
//                      // memory etc.
//   }
//   return block_val;
// }
//
// // Performs a _deterministic_ sum reduction. determinism is achieved by
// // requiring that only a single block be used.
// template <class Float>
// __global__ void global_sum_single_block_kernel(float *result,
//                                                const Float *values,
//                                                size_t count) {
//   assert(gridDim.x == 1); // only a single block!
//   float thread_sum = 0;
//   for (size_t index = threadIdx.x; index < count; index += blockDim.x) {
//     thread_sum += (float)values[index];
//   }
//
//   float reduction = blockReduce<warpReduceSum>(thread_sum, true);
//   if (threadIdx.x == 0) {
//     *result = reduction;
//   }
// }
//
// template <class Float>
// void global_sum_deterministic(float *result, const Float *values, int count,
//                               cudaStream_t stream) {
//   global_sum_single_block_kernel<<<1, 1024, 0, stream>>>(result, values,
//   count); cudaCheck(cudaGetLastError());
// }

#endif
