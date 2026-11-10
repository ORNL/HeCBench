#pragma once

#include <cub/warp/warp_reduce.cuh>

#include "types.h"

// True when any lane in this 32-lane group has pred set.
//
// On CUDA the group is the whole warp, so the full-warp mask names every
// lane. Every lane of the warp must reach this call.
__device__ inline bool logical_warp_any(bool pred) {
    return __ballot_sync(0xffffffffu, static_cast<int>(pred)) != 0u;
}

// Sum one float across the 32-lane group. Only lane 0 of the group is
// guaranteed to receive the full sum; the other lanes hold partial sums.
// Scratch is warp_storage_base[flat thread id / LOGICAL_WARP_SIZE]. The block
// is two-dimensional, so threadIdx.x alone would put every group in slot 0.
template <int LOGICAL_WARP_SIZE = kWaveSize>
__device__ inline void warpSum_scalar(
    float& val,
    typename cub::WarpReduce<float, LOGICAL_WARP_SIZE>::TempStorage* warp_storage_base) {
    using warp_reduce_t = cub::WarpReduce<float, LOGICAL_WARP_SIZE>;
    const int warp_id = (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) / LOGICAL_WARP_SIZE;
    warp_reduce_t wreduce(warp_storage_base[warp_id]);
    val = wreduce.Sum(val);
}

// Sum one float across the 32-lane group.
template <int LOGICAL_WARP_SIZE = kWaveSize>
__device__ inline void warpSum(
    float& x,
    typename cub::WarpReduce<float, LOGICAL_WARP_SIZE>::TempStorage* warp_storage_base) {
    warpSum_scalar<LOGICAL_WARP_SIZE>(x, warp_storage_base);
}

// Sum x and y independently across the 32-lane group.
template <int LOGICAL_WARP_SIZE = kWaveSize>
__device__ inline void warpSum(
    vec2& v,
    typename cub::WarpReduce<float, LOGICAL_WARP_SIZE>::TempStorage* warp_storage_base) {
    warpSum_scalar<LOGICAL_WARP_SIZE>(v.x, warp_storage_base);
    warpSum_scalar<LOGICAL_WARP_SIZE>(v.y, warp_storage_base);
}

// Sum x, y, and z independently across the 32-lane group.
template <int LOGICAL_WARP_SIZE = kWaveSize>
__device__ inline void warpSum(
    vec3& v,
    typename cub::WarpReduce<float, LOGICAL_WARP_SIZE>::TempStorage* warp_storage_base) {
    warpSum_scalar<LOGICAL_WARP_SIZE>(v.x, warp_storage_base);
    warpSum_scalar<LOGICAL_WARP_SIZE>(v.y, warp_storage_base);
    warpSum_scalar<LOGICAL_WARP_SIZE>(v.z, warp_storage_base);
}

// Sum each of the N channels independently across the 32-lane group.
template <int N, int LOGICAL_WARP_SIZE = kWaveSize>
__device__ inline void warpSum(
    float (&a)[N],
    typename cub::WarpReduce<float, LOGICAL_WARP_SIZE>::TempStorage* warp_storage_base) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
        warpSum_scalar<LOGICAL_WARP_SIZE>(a[i], warp_storage_base);
    }
}
