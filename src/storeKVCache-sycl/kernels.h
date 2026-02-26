#include <cstdint>

// Each warp will process 256 bytes per loop iteration
template <typename T, int WARP_SIZE=32>
void store_kv_cache_256x1(
    uint64_t* __restrict__ k_cache,
    uint64_t* __restrict__ v_cache,
    const T* __restrict__ out_loc,
    const uint64_t batch_size,
    const uint64_t* __restrict__ k,
    const uint64_t* __restrict__ v,
    const uint64_t kv_cache_stride,
    const uint64_t kv_input_stride,
    const uint64_t num_items,
    sycl::nd_item<1> &item)
{
  const auto idx = item.get_global_id(0);
  const auto warp_id = idx / WARP_SIZE;
  const auto lane_id = idx % WARP_SIZE;
  if (warp_id >= batch_size) return;

  const auto k_src = k + warp_id * kv_input_stride;
  const auto v_src = v + warp_id * kv_input_stride;

  const auto offset = out_loc[warp_id]; // [0, cache_size)
  const auto k_dst = k_cache + offset * kv_cache_stride;
  const auto v_dst = v_cache + offset * kv_cache_stride;

  for (uint64_t i = 0; i < num_items; ++i) {
    k_dst[lane_id + i * WARP_SIZE] = k_src[lane_id + i * WARP_SIZE];
    v_dst[lane_id + i * WARP_SIZE] = v_src[lane_id + i * WARP_SIZE];
  }
}

// Each warp will process 256x2 bytes per loop iteration
template <typename T, int WARP_SIZE=32>
void store_kv_cache_256x1_v2(
    uint64_t* __restrict__ k_cache,
    uint64_t* __restrict__ v_cache,
    const T* __restrict__ out_loc,
    const uint64_t batch_size,
    const uint64_t* __restrict__ k,
    const uint64_t* __restrict__ v,
    const uint64_t kv_cache_stride,
    const uint64_t kv_input_stride,
    const uint64_t num_items,
    sycl::nd_item<1> &item)
{
  const auto idx = item.get_global_id(0);
  const auto warp_id = idx / WARP_SIZE;
  const auto lane_id = idx % WARP_SIZE;
  if (warp_id >= batch_size) return;

  const auto k_src = k + warp_id * kv_input_stride;
  const auto v_src = v + warp_id * kv_input_stride;

  const auto offset = out_loc[warp_id]; // [0, cache_size)
  const auto k_dst = k_cache + offset * kv_cache_stride;
  const auto v_dst = v_cache + offset * kv_cache_stride;

  for (uint64_t i = 0; i < num_items; ++i) {
    reinterpret_cast<sycl::ulong2 *>(k_dst)[lane_id + i * WARP_SIZE] =
        reinterpret_cast<const sycl::ulong2 *>(k_src)[lane_id + i * WARP_SIZE];
    reinterpret_cast<sycl::ulong2 *>(v_dst)[lane_id + i * WARP_SIZE] =
        reinterpret_cast<const sycl::ulong2 *>(v_src)[lane_id + i * WARP_SIZE];
  }
}

// Each warp will process 256x4 bytes per loop iteration
template <typename T, int WARP_SIZE=32>
void store_kv_cache_256x1_v4(
    uint64_t* __restrict__ k_cache,
    uint64_t* __restrict__ v_cache,
    const T* __restrict__ out_loc,
    const uint64_t batch_size,
    const uint64_t* __restrict__ k,
    const uint64_t* __restrict__ v,
    const uint64_t kv_cache_stride,
    const uint64_t kv_input_stride,
    const uint64_t num_items,
    sycl::nd_item<1> &item)
{
  const auto idx = item.get_global_id(0);
  const auto warp_id = idx / WARP_SIZE;
  const auto lane_id = idx % WARP_SIZE;
  if (warp_id >= batch_size) return;

  const auto k_src = k + warp_id * kv_input_stride;
  const auto v_src = v + warp_id * kv_input_stride;

  const auto offset = out_loc[warp_id]; // [0, cache_size)
  const auto k_dst = k_cache + offset * kv_cache_stride;
  const auto v_dst = v_cache + offset * kv_cache_stride;

  for (uint64_t i = 0; i < num_items; ++i) {
    reinterpret_cast<sycl::ulong4 *>(k_dst)[lane_id + i * WARP_SIZE] =
        reinterpret_cast<const sycl::ulong4 *>(k_src)[lane_id + i * WARP_SIZE];
    reinterpret_cast<sycl::ulong4 *>(v_dst)[lane_id + i * WARP_SIZE] =
        reinterpret_cast<const sycl::ulong4 *>(v_src)[lane_id + i * WARP_SIZE];
  }
}

// Each warp will process 128 bytes per loop iteration
template <typename T, int WARP_SIZE=32>
void store_kv_cache_128x2(
    uint64_t* __restrict__ k_cache,
    uint64_t* __restrict__ v_cache,
    const T* __restrict__ out_loc,
    const uint64_t batch_size,
    const uint64_t* __restrict__ k,
    const uint64_t* __restrict__ v,
    const uint64_t kv_cache_stride,
    const uint64_t kv_input_stride,
    const uint64_t num_items,
    sycl::nd_item<1> &item)
{
  const auto idx = item.get_global_id(0);
  const auto warp_id = idx / WARP_SIZE;
  const auto lane_id = idx % WARP_SIZE;
  if (warp_id >= batch_size) return;
  const auto offset = out_loc[warp_id];

  const auto copy_k = lane_id < WARP_SIZE/2;
  const auto copy_id = lane_id % (WARP_SIZE/2);
  const auto cache = copy_k ? k_cache : v_cache;
  const auto input = copy_k ? k : v;
  const auto dst = cache + offset * kv_cache_stride;
  const auto src = input + warp_id * kv_input_stride;
  for (uint64_t i = 0; i < num_items; ++i) {
    dst[copy_id + i * (WARP_SIZE/2)] = src[copy_id + i * (WARP_SIZE/2)];
  }
}
