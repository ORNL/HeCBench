// Intra-node MoE combine, bf16 expert outputs, fp32 accumulation.
// Expert outputs are read through IPC-mapped pointers (the MORI P2P-read
// combine). peer_rendezvous is a one-block kernel: it publishes this rank's
// epoch and waits for every rank. The combine grid is launched after it.
// A spin inside the combine grid can deadlock, because the device does not
// promise to schedule the publishing block while other blocks are resident.
// Remote vector loads are issued before the weighted sum so their latency
// overlaps the arithmetic. A single rank skips the flag exchange.

#pragma once

#include <cstdint>
#include "reference.hpp"

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

#if defined(__HIPCC__)
#define MC_HD __host__ __device__
// Store one flag so other ranks can observe it.
__device__ inline void mc_store_u64(unsigned long long *p, unsigned long long v) {
  __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
// Load one flag published by another rank.
__device__ inline unsigned long long mc_load_u64(const unsigned long long *p) {
  return __hip_atomic_load(p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}
// Back off while a flag is not yet visible.
__device__ inline void mc_pause() { __builtin_amdgcn_s_sleep(4); }
#else
#define MC_HD __host__ __device__
// Store one flag so other ranks can observe it.
__device__ inline void mc_store_u64(unsigned long long *p, unsigned long long v) {
  atomicExch_system(p, v);
}
// Load one flag published by another rank.
__device__ inline unsigned long long mc_load_u64(const unsigned long long *p) {
  return atomicAdd_system(const_cast<unsigned long long *>(p), 0ull);
}
// Back off while a flag is not yet visible.
__device__ inline void mc_pause() { __nanosleep(64); }
#endif

// Expand a bf16 bit pattern to fp32.
MC_HD inline float mc_bf16_to_f32(std::uint16_t b) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return __uint_as_float(static_cast<unsigned>(b) << 16);
#else
  return bf16_to_f32(b);
#endif
}

// Round-to-nearest-even bf16, matching the host helper.
MC_HD inline std::uint16_t mc_f32_to_bf16(float x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  const unsigned bits = __float_as_uint(x);
  const unsigned lsb = (bits >> 16) & 1u;
  return static_cast<std::uint16_t>((bits + 0x7fffu + lsb) >> 16);
#else
  return f32_to_bf16(x);
#endif
}

// Load eight bf16 values as one 16-byte vector.
__device__ inline uint4 load_raw8(const std::uint16_t *p) {
  return *reinterpret_cast<const uint4 *>(p);
}

// Unpack eight bf16 values to fp32.
__device__ inline void decode8(uint4 v, float *o) {
  const unsigned w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    o[2 * i] = mc_bf16_to_f32(static_cast<std::uint16_t>(w[i]));
    o[2 * i + 1] = mc_bf16_to_f32(static_cast<std::uint16_t>(w[i] >> 16));
  }
}

// Pack eight fp32 values and store them as bf16.
__device__ inline void store8(std::uint16_t *p, const float *a) {
  unsigned w[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const unsigned lo = mc_f32_to_bf16(a[2 * i]);
    const unsigned hi = mc_f32_to_bf16(a[2 * i + 1]);
    w[i] = lo | (hi << 16);
  }
  *reinterpret_cast<uint4 *>(p) = make_uint4(w[0], w[1], w[2], w[3]);
}

// One block publishes this epoch and waits until every rank has published
// this epoch or a later one. Ranks do not barrier between iterations, so a
// peer can already hold a newer epoch; equality would miss it and spin.
__global__ void peer_rendezvous(unsigned long long *local_flags,
                                unsigned long long *const *peer_flags, int rank, int world,
                                unsigned long long epoch) {
  if (threadIdx.x != 0)
    return;
  __threadfence_system();
  for (int peer = 0; peer < world; ++peer)
    mc_store_u64(peer_flags[peer] + rank, epoch);
  __threadfence_system();
  for (int peer = 0; peer < world; ++peer)
    while (mc_load_u64(local_flags + peer) < epoch)
      mc_pause();
  __threadfence_system();
}

// Weighted bf16 combine. The peer rendezvous has already completed.
template <int kTopk>
__global__ void __launch_bounds__(256)
    combine_kernel(int tokens, int hidden, const int *__restrict__ dest,
                   const int *__restrict__ slot,
                   const std::uint16_t *const *__restrict__ stage,
                   std::uint16_t *__restrict__ out) {
  const int nvec = hidden >> 3;
  for (int token = blockIdx.x; token < tokens; token += gridDim.x) {
    for (int vec = threadIdx.x; vec < nvec; vec += blockDim.x) {
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      const int *td = dest + static_cast<std::size_t>(token) * kTopk;
      const int *ts = slot + static_cast<std::size_t>(token) * kTopk;
      uint4 raw[kTopk];
      // A negative destination is not read. Dispatch keeps one expert per
      // destination rank, and a top-k larger than the pool leaves the rest empty.
#pragma unroll
      for (int k = 0; k < kTopk; ++k) {
        if (td[k] < 0)
          continue;
        const std::uint16_t *src =
            stage[td[k]] +
            (static_cast<std::size_t>(ts[k]) * hidden +
             (static_cast<std::size_t>(vec) << 3));
        raw[k] = load_raw8(src);
      }
#pragma unroll
      for (int k = 0; k < kTopk; ++k) {
        if (td[k] < 0)
          continue;
        float x[8];
        decode8(raw[k], x);
        const float w = weight_of(token, k, kTopk);
#pragma unroll
        for (int j = 0; j < 8; ++j)
          acc[j] += w * x[j];
      }
      store8(out + static_cast<std::size_t>(token) * hidden +
                 (static_cast<std::size_t>(vec) << 3),
             acc);
    }
  }
}

// Write this rank's deterministic expert outputs.
__global__ void fill_stage(std::uint16_t *stage, int rank, int slots, int hidden) {
  const std::size_t n = static_cast<std::size_t>(slots) * hidden;
  for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const int slot = static_cast<int>(i / hidden);
    const int h = static_cast<int>(i - static_cast<std::size_t>(slot) * hidden);
    stage[i] = mc_f32_to_bf16(stage_unit(rank, slot, h));
  }
}
