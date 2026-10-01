// Torch/ATen replacement for the standalone TwELL HIP benchmark.
//
// This is the HIP (ROCm) counterpart of the CUDA torch_compat.h. The TwELL
// kernels reference `at::BFloat16` purely as a 2-byte storage type at the host
// API boundary and immediately reinterpret_cast it to the native bf16 type
// inside the kernels. Instead of pulling in libtorch/ATen, we alias
// `at::BFloat16` to HIP's own bf16 type (`__hip_bfloat16`).
#pragma once

#include <cstdint>

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

namespace at {

// Replace the ATen bf16 type with the native HIP bf16 type.
using BFloat16 = __hip_bfloat16;

static_assert(sizeof(BFloat16) == 2, "HIP __hip_bfloat16 must be 2 bytes");
static_assert(alignof(BFloat16) == 2, "HIP __hip_bfloat16 must be 2-byte aligned");

}  // namespace at

// ---------------------------------------------------------------------------
// CUDA type-name aliases.
//
// The ported kernels keep the original `__nv_bfloat16` / `__nv_bfloat162`
// spellings; map them onto the HIP types so the bodies compile unchanged.
using __nv_bfloat16 = __hip_bfloat16;
using __nv_bfloat162 = __hip_bfloat162;

// ---------------------------------------------------------------------------
// CUDA intrinsic shims not provided by HIP.

// Round-to-nearest float->bf16. HIP's __float2bfloat16 already rounds to
// nearest-even, so this is a straight forward.
__device__ __host__ static inline __hip_bfloat16 __float2bfloat16_rn(float f) {
    return __float2bfloat16(f);
}

// Pack two floats into a bf16x2 (round-to-nearest-even each lane).
__device__ __host__ static inline __hip_bfloat162 __floats2bfloat162_rn(float a,
                                                                        float b) {
    return __halves2bfloat162(__float2bfloat16(a), __float2bfloat16(b));
}

// Streaming (cache-non-temporal) global load/store. HIP has no direct __ldcs /
// __stcs; a plain dereference is functionally equivalent (the streaming hint is
// only a cache-policy optimization, not a correctness requirement).
template <class T>
__device__ __forceinline__ T __ldcs(const T* ptr) {
    return *ptr;
}

template <class T>
__device__ __forceinline__ void __stcs(T* ptr, T value) {
    *ptr = value;
}
