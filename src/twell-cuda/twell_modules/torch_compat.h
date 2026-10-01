// Torch/ATen replacement for the standalone TwELL CUDA benchmark.
//
// The TwELL kernels reference `at::BFloat16` purely as a 2-byte storage type at
// the host API boundary and immediately `reinterpret_cast` it to the native
// CUDA `__nv_bfloat16` inside the kernels. Instead of pulling in libtorch/ATen,
// we simply alias `at::BFloat16` to CUDA's own bf16 type. This removes the ATen
// (and c10) dependency entirely: the benchmark is built with nothing but the
// CUDA toolkit, and every `at::BFloat16` in the kernels *is* a `__nv_bfloat16`.
//
// This header replaces what used to be provided by the external `stubs/`
// directory (stubs/ATen/ATen.h and stubs/c10/cuda/CUDAStream.h), keeping this
// kernel copy fully self-contained.
#pragma once

#include <cstdint>
#include <cuda_bf16.h>

// ---------------------------------------------------------------------------
// Minimum CUDA architecture support.
#ifndef TWELL_MIN_ARCH
#define TWELL_MIN_ARCH 900
#endif

// Compile-time guard: reject device compilation for any target below the
// minimum. __CUDA_ARCH__ is only defined during the device pass, so host-only
// tooling is unaffected.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < TWELL_MIN_ARCH)
#error "TwELL requires a CUDA architecture >= TWELL_MIN_ARCH (default sm_90a). \
Rebuild with a suitable -gencode (e.g. arch=compute_90a,code=sm_90a) or \
override TWELL_MIN_ARCH."
#endif

// Require the sm_90a feature set specifically: a plain sm_90 build has the same
// __CUDA_ARCH__ (900) but lacks WGMMA/TMA, so it would otherwise fail later
// with cryptic PTX errors. nvcc defines __CUDA_ARCH_FEAT_SM90_ALL only for the
// "a" variant. This check applies when targeting exactly sm_90.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 900) && \
    !defined(__CUDA_ARCH_FEAT_SM90_ALL)
#error "TwELL requires the sm_90a architecture-specific target for WGMMA/TMA. \
Rebuild with -gencode arch=compute_90a,code=sm_90a (not plain compute_90)."
#endif

namespace at {

// Replace the ATen bf16 type with the native CUDA bf16 type.
using BFloat16 = __nv_bfloat16;

static_assert(sizeof(BFloat16) == 2, "CUDA __nv_bfloat16 must be 2 bytes");
static_assert(alignof(BFloat16) == 2, "CUDA __nv_bfloat16 must be 2-byte aligned");

}  // namespace at

// ---------------------------------------------------------------------------
// Atomic builtin compatibility shim.
//
// matmul_d2t.cu uses the SM90 memory-model builtin `__nv_atomic_fetch_add`
// with `__NV_ATOMIC_RELAXED` / `__NV_THREAD_SCOPE_BLOCK`. These are provided
// natively by CUDA 13.0+ (the toolkit the upstream torch build targets). On
// older toolkits (e.g. CUDA 12.x) they are absent, so we supply a functionally
// equivalent implementation. The single call site operates on a uint32_t
// counter in shared memory, where a relaxed block-scoped fetch-add is exactly
// `atomicAdd`.
#if defined(__CUDACC__) && defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ < 13)

#ifndef __NV_ATOMIC_RELAXED
#define __NV_ATOMIC_RELAXED 0
#endif
#ifndef __NV_THREAD_SCOPE_BLOCK
#define __NV_THREAD_SCOPE_BLOCK 2
#endif

template <class T>
__device__ __forceinline__ T __nv_atomic_fetch_add(T* address, T value,
                                                    int /*memorder*/,
                                                    int /*scope*/) {
    return atomicAdd(address, value);
}

#endif  // CUDA < 13 atomic shim
