#pragma once

#include <sycl/sycl.hpp>

#include "types.h"

// Every helper here works on one 32-lane group inside a sub-group of 32 or 64
// lanes. A group is sub-group lanes 0-31 or 32-63, counted by sub-group lane
// id, not by work-group id. An xor partner below kWaveSize stays inside the
// caller's group. On a 64-lane sub-group the two halves therefore never
// exchange values, and they may reach these calls with different trip counts,
// as the two halves of a 64-wide HIP wave do.

// Lane of this work-item within its 32-lane group.
inline int logical_warp_lane(const sycl::sub_group& sg) {
    return sg.get_local_linear_id() & (kWaveSize - 1);
}

// Index of this work-item's 32-lane group within the work-group.
inline int logical_warp_id(const sycl::sub_group& sg) {
    return (sg.get_group_linear_id() * sg.get_max_local_range()[0] + sg.get_local_linear_id()) / kWaveSize;
}

// Placeholder scratch type. It keeps the shared-memory layout identical to the
// HIP and CUDA ports, whose warp reductions take one scratch slot per group.
// The shuffles below do not read it.
struct warp_reduce_storage {};

// Relaxed device-scope float add into global memory.
// On NVIDIA this is the red.add.f32 that CUDA's atomicAdd compiles to, and
// like it the result flushes subnormals. DPC++ lowers atomic_ref fetch_add
// there to a compare-and-swap loop unless the kernel flushes subnormals.
inline void atomicAdd(float* addr, float val) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    asm volatile("red.add.f32 [%0], %1;" ::"l"(addr), "f"(val) : "memory");
#else
    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>(*addr)
        .fetch_add(val);
#endif
}

// Relaxed work-group-scope float add into local memory.
// On Hopper, atomic_ref lowers to a compare-and-swap loop. The cluster
// reduction is one SM-local instruction for block local memory, matching the
// CUDA port. On AMD, atomic_ref already emits ds_add_f32, as the HIP port does.
// The Hopper instruction flushes subnormals.
inline void atomicAdd_block(float* addr, float val) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__) && (__CUDA_ARCH__ >= 900)
    unsigned p;
    asm volatile("cvta.to.shared.u32 %0, %1;" : "=r"(p) : "l"(addr));
    asm volatile("red.relaxed.cta.shared::cluster.add.f32 [%0], %1;" ::"r"(p), "f"(val));
#else
    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                     sycl::access::address_space::local_space>(*addr)
        .fetch_add(val);
#endif
}

// True when any lane in this 32-lane group has pred set.
// AMD and NVIDIA use a hardware ballot, as the HIP and CUDA ports do; it runs
// once per Gaussian, and the shuffle fallback costs five dependent round
// trips. sycl::ext::oneapi::group_ballot is not used because on a 64-lane
// AMD sub-group its upper 32 bits do not hold the upper half's votes.
inline bool logical_warp_any(const sycl::sub_group& sg, bool pred) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__)
    const uint64_t bits = __builtin_amdgcn_ballot_w64(pred);
    const int group = sg.get_local_linear_id() - logical_warp_lane(sg);
    return ((bits >> group) & 0xffffffffull) != 0ull;
#elif defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    (void)sg;
    return __nvvm_vote_ballot_sync(0xffffffffu, pred) != 0u;
#else
    int v = pred ? 1 : 0;
#pragma unroll
    for (int mask = kWaveSize / 2; mask > 0; mask >>= 1) {
        v |= sycl::permute_group_by_xor(sg, v, mask);
    }
    return v != 0;
#endif
}

// Maximum of val over this 32-lane group. Every lane receives it.
inline int logical_warp_max(const sycl::sub_group& sg, int val) {
#pragma unroll
    for (int mask = 1; mask < kWaveSize; mask <<= 1) {
        val = sycl::max(val, sycl::permute_group_by_xor(sg, val, mask));
    }
    return val;
}

// Sum one float across the 32-lane group. Only lane 0 receives the full sum.
// The steps pull from lanes 1, 2, 4, 8, and 16, which is the shuffle-down of
// the CUDA cub reduction and the HIP rocprim reduction, so lane 0 adds in the
// same order. On NVIDIA the add is predicated by the shuffle, as in cub. On
// AMD an out-of-range lane reads itself and adds that, as rocprim does; lane
// 0 never reads that doubled value.
template <int LOGICAL_WARP_SIZE = kWaveSize>
inline void warpSum_scalar(const sycl::sub_group& sg, float& val, warp_reduce_storage* warp_storage_base) {
    (void)warp_storage_base;
#if defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    (void)sg;
#pragma unroll
    for (int offset = 1; offset < LOGICAL_WARP_SIZE; offset <<= 1) {
        float result;
        asm volatile(
            "{"
            "  .reg .f32 r0;"
            "  .reg .pred p;"
            "  shfl.sync.down.b32 r0|p, %1, %2, %3, %4;"
            "  @p add.f32 r0, r0, %1;"
            "  mov.f32 %0, r0;"
            "}"
            : "=f"(result)
            : "f"(val), "r"(offset), "r"(LOGICAL_WARP_SIZE - 1), "r"(0xffffffff));
        val = result;
    }
#elif defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__)
    const int lane = static_cast<int>(sg.get_local_linear_id());
#pragma unroll
    for (int offset = 1; offset < LOGICAL_WARP_SIZE; offset <<= 1) {
        const int sub = lane & (LOGICAL_WARP_SIZE - 1);
        const int src = lane + ((sub + offset < LOGICAL_WARP_SIZE) ? offset : 0);
        const int got = __builtin_amdgcn_ds_bpermute(src << 2, sycl::bit_cast<int>(val));
        val += sycl::bit_cast<float>(got);
    }
#else
#pragma unroll
    for (int mask = 1; mask < LOGICAL_WARP_SIZE; mask <<= 1) {
        val += sycl::permute_group_by_xor(sg, val, mask);
    }
#endif
}

// Sum one float across the 32-lane group.
template <int LOGICAL_WARP_SIZE = kWaveSize>
inline void warpSum(const sycl::sub_group& sg, float& x, warp_reduce_storage* warp_storage_base) {
    warpSum_scalar<LOGICAL_WARP_SIZE>(sg, x, warp_storage_base);
}

// Sum x and y independently across the 32-lane group.
template <int LOGICAL_WARP_SIZE = kWaveSize>
inline void warpSum(const sycl::sub_group& sg, vec2& v, warp_reduce_storage* warp_storage_base) {
    warpSum_scalar<LOGICAL_WARP_SIZE>(sg, v.x, warp_storage_base);
    warpSum_scalar<LOGICAL_WARP_SIZE>(sg, v.y, warp_storage_base);
}

// Sum x, y, and z independently across the 32-lane group.
template <int LOGICAL_WARP_SIZE = kWaveSize>
inline void warpSum(const sycl::sub_group& sg, vec3& v, warp_reduce_storage* warp_storage_base) {
    warpSum_scalar<LOGICAL_WARP_SIZE>(sg, v.x, warp_storage_base);
    warpSum_scalar<LOGICAL_WARP_SIZE>(sg, v.y, warp_storage_base);
    warpSum_scalar<LOGICAL_WARP_SIZE>(sg, v.z, warp_storage_base);
}

// Sum each of the N channels independently across the 32-lane group.
template <int N, int LOGICAL_WARP_SIZE = kWaveSize>
inline void warpSum(const sycl::sub_group& sg, float (&a)[N], warp_reduce_storage* warp_storage_base) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
        warpSum_scalar<LOGICAL_WARP_SIZE>(sg, a[i], warp_storage_base);
    }
}
