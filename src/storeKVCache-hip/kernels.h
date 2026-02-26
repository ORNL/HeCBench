#include <cstdint>

// Each warp will process 256 bytes per loop iteration
template <typename T>
__global__ void store_kv_cache_256x1(
    uint64_t* __restrict__ k_cache,
    uint64_t* __restrict__ v_cache,
    const T* __restrict__ out_loc,
    const uint64_t batch_size,
    const uint64_t* __restrict__ k,
    const uint64_t* __restrict__ v,
    const uint64_t kv_cache_stride,
    const uint64_t kv_input_stride,
    const uint64_t num_items)
{
#if defined(__GFX8__) || defined(__GFX9__)
  #define WarpSize 64
#else
  #define WarpSize 32
#endif
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto warp_id = idx / WarpSize;
  const auto lane_id = idx % WarpSize;
  if (warp_id >= batch_size) return;

  const auto k_src = k + warp_id * kv_input_stride;
  const auto v_src = v + warp_id * kv_input_stride;

  const auto offset = out_loc[warp_id]; // [0, cache_size)
  const auto k_dst = k_cache + offset * kv_cache_stride;
  const auto v_dst = v_cache + offset * kv_cache_stride;

  for (uint64_t i = 0; i < num_items; ++i) {
    k_dst[lane_id + i * WarpSize] = k_src[lane_id + i * WarpSize];
    v_dst[lane_id + i * WarpSize] = v_src[lane_id + i * WarpSize];
  }
}

// Each warp will process 256x2 bytes per loop iteration
template <typename T>
__global__ void store_kv_cache_256x1_v2(
    uint64_t* __restrict__ k_cache,
    uint64_t* __restrict__ v_cache,
    const T* __restrict__ out_loc,
    const uint64_t batch_size,
    const uint64_t* __restrict__ k,
    const uint64_t* __restrict__ v,
    const uint64_t kv_cache_stride,
    const uint64_t kv_input_stride,
    const uint64_t num_items)
{
#if defined(__GFX8__) || defined(__GFX9__)
  #define WarpSize 64
#else
  #define WarpSize 32
#endif
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto warp_id = idx / WarpSize;
  const auto lane_id = idx % WarpSize;
  if (warp_id >= batch_size) return;

  const auto k_src = k + warp_id * kv_input_stride;
  const auto v_src = v + warp_id * kv_input_stride;

  const auto offset = out_loc[warp_id]; // [0, cache_size)
  const auto k_dst = k_cache + offset * kv_cache_stride;
  const auto v_dst = v_cache + offset * kv_cache_stride;

  for (uint64_t i = 0; i < num_items; ++i) {
    reinterpret_cast<ulong2*>(k_dst)[lane_id + i * WarpSize] =
      reinterpret_cast<const ulong2*>(k_src)[lane_id + i * WarpSize];
    reinterpret_cast<ulong2*>(v_dst)[lane_id + i * WarpSize] =
      reinterpret_cast<const ulong2*>(v_src)[lane_id + i * WarpSize];
  }
}

// Each warp will process 256x4 bytes per loop iteration
template <typename T>
__global__ void store_kv_cache_256x1_v4(
    uint64_t* __restrict__ k_cache,
    uint64_t* __restrict__ v_cache,
    const T* __restrict__ out_loc,
    const uint64_t batch_size,
    const uint64_t* __restrict__ k,
    const uint64_t* __restrict__ v,
    const uint64_t kv_cache_stride,
    const uint64_t kv_input_stride,
    const uint64_t num_items)
{
#if defined(__GFX8__) || defined(__GFX9__)
  #define WarpSize 64
#else
  #define WarpSize 32
#endif
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto warp_id = idx / WarpSize;
  const auto lane_id = idx % WarpSize;
  if (warp_id >= batch_size) return;

  const auto k_src = k + warp_id * kv_input_stride;
  const auto v_src = v + warp_id * kv_input_stride;

  const auto offset = out_loc[warp_id]; // [0, cache_size)
  const auto k_dst = k_cache + offset * kv_cache_stride;
  const auto v_dst = v_cache + offset * kv_cache_stride;

  for (uint64_t i = 0; i < num_items; ++i) {
    reinterpret_cast<ulong4*>(k_dst)[lane_id + i * WarpSize] =
      reinterpret_cast<const ulong4*>(k_src)[lane_id + i * WarpSize];
    reinterpret_cast<ulong4*>(v_dst)[lane_id + i * WarpSize] =
      reinterpret_cast<const ulong4*>(v_src)[lane_id + i * WarpSize];
  }
}

// Each warp will process 128 bytes per loop iteration
template <typename T>
__global__ void store_kv_cache_128x2(
    uint64_t* __restrict__ k_cache,
    uint64_t* __restrict__ v_cache,
    const T* __restrict__ out_loc,
    const uint64_t batch_size,
    const uint64_t* __restrict__ k,
    const uint64_t* __restrict__ v,
    const uint64_t kv_cache_stride,
    const uint64_t kv_input_stride,
    const uint64_t num_items)
{
#if defined(__GFX8__) || defined(__GFX9__)
  #define WarpSize 64
#else
  #define WarpSize 32
#endif
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  const auto warp_id = idx / WarpSize;
  const auto lane_id = idx % WarpSize;
  if (warp_id >= batch_size) return;
  const auto offset = out_loc[warp_id];

  const auto copy_k = lane_id < WarpSize/2;
  const auto copy_id = lane_id % (WarpSize/2);
  const auto cache = copy_k ? k_cache : v_cache;
  const auto input = copy_k ? k : v;
  const auto dst = cache + offset * kv_cache_stride;
  const auto src = input + warp_id * kv_input_stride;
  for (uint64_t i = 0; i < num_items; ++i) {
    dst[copy_id + i * (WarpSize/2)] = src[copy_id + i * (WarpSize/2)];
  }
}
