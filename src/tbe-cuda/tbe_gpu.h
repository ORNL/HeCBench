// CUDA/HIP kernels inspired by FBGEMM's BSD-3-Clause licensed public
// SplitTableBatchedEmbeddingBags implementation (https://github.com/pytorch/FBGEMM).
// This dependency-free implementation uses the same central optimization idea:
// one warp/wavefront per pooled row and vectorized processing for common D.

#pragma once

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define TBE_GPU_CHECK(call)                                                    \
  do {                                                                         \
    hipError_t e_ = (call);                                                     \
    if (e_ != hipSuccess) {                                                     \
      std::fprintf(stderr, "HIP error: %s\n", hipGetErrorString(e_));           \
      std::exit(2);                                                             \
    }                                                                          \
  } while (0)
#define TBE_MALLOC hipMalloc
#define TBE_FREE hipFree
#define TBE_MEMCPY hipMemcpy
#define TBE_H2D hipMemcpyHostToDevice
#define TBE_D2H hipMemcpyDeviceToHost
#define TBE_MEMSET hipMemset
#define TBE_SYNC hipDeviceSynchronize
#define TBE_LAST_ERROR hipGetLastError
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define TBE_GPU_CHECK(call)                                                    \
  do {                                                                         \
    cudaError_t e_ = (call);                                                    \
    if (e_ != cudaSuccess) {                                                    \
      std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e_));         \
      std::exit(2);                                                             \
    }                                                                          \
  } while (0)
#define TBE_MALLOC cudaMalloc
#define TBE_FREE cudaFree
#define TBE_MEMCPY cudaMemcpy
#define TBE_H2D cudaMemcpyHostToDevice
#define TBE_D2H cudaMemcpyDeviceToHost
#define TBE_MEMSET cudaMemset
#define TBE_SYNC cudaDeviceSynchronize
#define TBE_LAST_ERROR cudaGetLastError
#endif

#include "tbe_common.h"

// One warp or wavefront owns one pooled row. D of 64, 128, 192, or 256
// loads four FP16 values, accumulates them in FP32, and stores FP16.
// D of 0 is the scalar path. Lane count is warpSize.
struct alignas(8) TbeHalf4 {
  __half x, y, z, w;
};

__device__ inline void tbe_atomic_add(TbeHalf* addr, __half value) {
  __half* p = reinterpret_cast<__half*>(addr);
#if defined(__HIPCC__)
  unsafeAtomicAdd(p, value);
#else
  atomicAdd(p, value);
#endif
}

template <int D>
__global__ void tbe_forward_kernel(
    const TbeHalf* __restrict__ weights,
    const int* __restrict__ indices,
    TbeHalf* __restrict__ output,
    int bags,
    int batch,
    int rows,
    int bag_size,
    int dim) {
  const int lane = threadIdx.x % warpSize;
  const int warp_in_block = threadIdx.x / warpSize;
  const int warps_per_block = blockDim.x / warpSize;
  const int bag_id = blockIdx.x * warps_per_block + warp_in_block;
  if (bag_id >= bags) return;
  const size_t bag = bag_id;
  const size_t table = bag / batch;
  const size_t index_base = bag * bag_size;
  const size_t output_base = bag * dim;
  const __half* w16 = reinterpret_cast<const __half*>(weights);
  __half* o16 = reinterpret_cast<__half*>(output);

  if constexpr (D != 0) {
    TbeHalf4* out4 = reinterpret_cast<TbeHalf4*>(o16 + output_base);
    for (int v = lane; v < D / 4; v += warpSize) {
      float4 sum = {0.f, 0.f, 0.f, 0.f};
      for (int l = 0; l < bag_size; ++l) {
        const int row = indices[index_base + l];
        const size_t w = (table * rows + row) * D;
        const TbeHalf4 x = reinterpret_cast<const TbeHalf4*>(w16 + w)[v];
        sum.x += __half2float(x.x);
        sum.y += __half2float(x.y);
        sum.z += __half2float(x.z);
        sum.w += __half2float(x.w);
      }
      out4[v] = TbeHalf4{__float2half(sum.x), __float2half(sum.y), __float2half(sum.z),
                         __float2half(sum.w)};
    }
  } else {
    for (int d = lane; d < dim; d += warpSize) {
      float sum = 0.f;
      for (int l = 0; l < bag_size; ++l) {
        const int row = indices[index_base + l];
        const size_t w = (table * rows + row) * dim;
        sum += __half2float(w16[w + d]);
      }
      o16[output_base + d] = __float2half(sum);
    }
  }
}

// Scatter-add the pooled gradient into every looked-up row. Repeated indices
// in one bag each issue their own atomic; there is no run-length cache.
template <int D>
__global__ void tbe_backward_kernel(
    const int* __restrict__ indices,
    const TbeHalf* __restrict__ grad_output,
    TbeHalf* __restrict__ grad_weight,
    int bags,
    int batch,
    int rows,
    int bag_size,
    int dim) {
  const int lane = threadIdx.x % warpSize;
  const int warp_in_block = threadIdx.x / warpSize;
  const int warps_per_block = blockDim.x / warpSize;
  const int bag_id = blockIdx.x * warps_per_block + warp_in_block;
  if (bag_id >= bags) return;
  const size_t bag = bag_id;
  const size_t table = bag / batch;
  const size_t index_base = bag * bag_size;
  const size_t grad_base = bag * dim;
  const __half* g16 = reinterpret_cast<const __half*>(grad_output);

  if constexpr (D != 0) {
    const TbeHalf4* grad4 = reinterpret_cast<const TbeHalf4*>(g16 + grad_base);
    for (int v = lane; v < D / 4; v += warpSize) {
      const TbeHalf4 g = grad4[v];
      for (int l = 0; l < bag_size; ++l) {
        const int row = indices[index_base + l];
        const size_t w = (table * rows + row) * D + 4 * v;
        tbe_atomic_add(grad_weight + w, g.x);
        tbe_atomic_add(grad_weight + w + 1, g.y);
        tbe_atomic_add(grad_weight + w + 2, g.z);
        tbe_atomic_add(grad_weight + w + 3, g.w);
      }
    }
  } else {
    for (int d = lane; d < dim; d += warpSize) {
      const __half g = g16[grad_base + d];
      for (int l = 0; l < bag_size; ++l) {
        const int row = indices[index_base + l];
        const size_t w = (table * rows + row) * dim + d;
        tbe_atomic_add(grad_weight + w, g);
      }
    }
  }
}

// Host lane count for one wave in the grid. CUDA warps are 32 threads.
// HIP queries the device wave size, which is 32 or 64.
inline int tbe_grid_lanes() {
#if defined(__HIPCC__)
  int lanes = 0;
  TBE_GPU_CHECK(hipDeviceGetAttribute(&lanes, hipDeviceAttributeWarpSize, 0));
  if (lanes <= 0 || tbe_block_threads % lanes != 0) {
    std::fprintf(stderr, "Unsupported warp size %d\n", lanes);
    std::exit(2);
  }
  return lanes;
#else
  return 32;
#endif
}

template <int D>
inline void tbe_launch_forward_fixed(
    const TBEProblem& p, const TbeHalf* w, const int* i, TbeHalf* o, int lanes) {
  tbe_forward_kernel<D><<<tbe_grid_blocks(p.bags(), lanes), tbe_block_threads>>>(
      w, i, o, p.bags(), p.batch, p.rows, p.bag, p.dim);
}

// Same grid as the forward launch: one wave per pooled row.
template <int D>
inline void tbe_launch_backward_fixed(
    const TBEProblem& p, const int* i, const TbeHalf* go, TbeHalf* gw, int lanes) {
  tbe_backward_kernel<D><<<tbe_grid_blocks(p.bags(), lanes), tbe_block_threads>>>(
      i, go, gw, p.bags(), p.batch, p.rows, p.bag, p.dim);
}

// Select the vectorized kernel for the published embedding widths.
inline void tbe_launch_forward(
    const TBEProblem& p, const TbeHalf* w, const int* i, TbeHalf* o, int lanes) {
  switch (p.dim) {
    case 64: tbe_launch_forward_fixed<64>(p, w, i, o, lanes); break;
    case 128: tbe_launch_forward_fixed<128>(p, w, i, o, lanes); break;
    case 192: tbe_launch_forward_fixed<192>(p, w, i, o, lanes); break;
    case 256: tbe_launch_forward_fixed<256>(p, w, i, o, lanes); break;
    default: tbe_launch_forward_fixed<0>(p, w, i, o, lanes);
  }
}

// Select the matching backward kernel for the same widths.
inline void tbe_launch_backward(
    const TBEProblem& p, const int* i, const TbeHalf* go, TbeHalf* gw, int lanes) {
  switch (p.dim) {
    case 64: tbe_launch_backward_fixed<64>(p, i, go, gw, lanes); break;
    case 128: tbe_launch_backward_fixed<128>(p, i, go, gw, lanes); break;
    case 192: tbe_launch_backward_fixed<192>(p, i, go, gw, lanes); break;
    case 256: tbe_launch_backward_fixed<256>(p, i, go, gw, lanes); break;
    default: tbe_launch_backward_fixed<0>(p, i, go, gw, lanes);
  }
}
