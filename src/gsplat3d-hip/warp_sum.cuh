#pragma once

#include <rocprim/warp/warp_reduce.hpp>

#include "types.h"

// True when any lane in this 32-lane group has pred set.
//
// cooperative_groups tile::any() calls __any_sync. That helper requires the
// mask it is given to name every active lane in the hardware wave. A 32-lane
// tile on a 64-wide wave only names half of the wave, so the check fails and,
// with device asserts enabled, executes a trap. The ballot here stays inside
// the caller's group. On a 32-wide wave the group is the whole wave, so the
// mask covers every lane.
__device__ inline bool logical_warp_any(bool pred) {
    const unsigned long long bits = __ballot(static_cast<int>(pred));
    const unsigned group = __lane_id() & ~(static_cast<unsigned>(kWaveSize) - 1u);
    const unsigned long long mask = ((1ull << kWaveSize) - 1ull) << group;
    return (bits & mask) != 0ull;
}

// Sum one float across the 32-lane group. Only lane 0 of the group is
// guaranteed to receive the full sum; the other lanes hold partial sums.
// Scratch is warp_storage_base[flat thread id / LOGICAL_WARP_SIZE]. The block
// is two-dimensional, so threadIdx.x alone would put every group in slot 0.
template <int LOGICAL_WARP_SIZE = kWaveSize>
__device__ inline void warpSum_scalar(
    float& val,
    typename rocprim::warp_reduce<float, LOGICAL_WARP_SIZE>::storage_type* warp_storage_base) {
    using warp_reduce_t = rocprim::warp_reduce<float, LOGICAL_WARP_SIZE>;
    const int warp_id = (threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z)) / LOGICAL_WARP_SIZE;
    warp_reduce_t wreduce;
    float sum;
    wreduce.reduce(val, sum, warp_storage_base[warp_id], rocprim::plus<float>());
    val = sum;
}

// Sum one float across the 32-lane group.
template <int LOGICAL_WARP_SIZE = kWaveSize>
__device__ inline void warpSum(
    float& x,
    typename rocprim::warp_reduce<float, LOGICAL_WARP_SIZE>::storage_type* warp_storage_base) {
    warpSum_scalar<LOGICAL_WARP_SIZE>(x, warp_storage_base);
}

// Sum x and y independently across the 32-lane group.
template <int LOGICAL_WARP_SIZE = kWaveSize>
__device__ inline void warpSum(
    vec2& v,
    typename rocprim::warp_reduce<float, LOGICAL_WARP_SIZE>::storage_type* warp_storage_base) {
    warpSum_scalar<LOGICAL_WARP_SIZE>(v.x, warp_storage_base);
    warpSum_scalar<LOGICAL_WARP_SIZE>(v.y, warp_storage_base);
}

// Sum x, y, and z independently across the 32-lane group.
template <int LOGICAL_WARP_SIZE = kWaveSize>
__device__ inline void warpSum(
    vec3& v,
    typename rocprim::warp_reduce<float, LOGICAL_WARP_SIZE>::storage_type* warp_storage_base) {
    warpSum_scalar<LOGICAL_WARP_SIZE>(v.x, warp_storage_base);
    warpSum_scalar<LOGICAL_WARP_SIZE>(v.y, warp_storage_base);
    warpSum_scalar<LOGICAL_WARP_SIZE>(v.z, warp_storage_base);
}

// Sum each of the N channels independently across the 32-lane group.
template <int N, int LOGICAL_WARP_SIZE = kWaveSize>
__device__ inline void warpSum(
    float (&a)[N],
    typename rocprim::warp_reduce<float, LOGICAL_WARP_SIZE>::storage_type* warp_storage_base) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
        warpSum_scalar<LOGICAL_WARP_SIZE>(a[i], warp_storage_base);
    }
}
