// SYCL counterpart of moe-combine-cuda/combine.cuh. Function names match that
// header: mc_bf16_to_f32, mc_f32_to_bf16, load_raw8, decode8, store8,
// peer_rendezvous, combine_kernel, and fill_stage.

#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

#include "reference.hpp"

// Round-to-nearest-even bf16.
inline std::uint16_t mc_f32_to_bf16(float x) {
  const std::uint32_t bits0 = sycl::bit_cast<std::uint32_t>(x);
  const std::uint32_t lsb = (bits0 >> 16) & 1u;
  return static_cast<std::uint16_t>((bits0 + 0x7fffu + lsb) >> 16);
}

// Expand a bf16 bit pattern to fp32.
inline float mc_bf16_to_f32(std::uint16_t b) {
  return sycl::bit_cast<float>(static_cast<std::uint32_t>(b) << 16);
}

using sys_atomic_u64 =
    sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed,
                     sycl::memory_scope::system, sycl::access::address_space::global_space>;

// Store one flag so other ranks can observe it.
inline void mc_store_u64(unsigned long long *p, unsigned long long v) {
  sys_atomic_u64(*p).store(v);
}

// Load one flag published by another rank.
inline unsigned long long mc_load_u64(unsigned long long *p) {
  return sys_atomic_u64(*p).load();
}

// No pause intrinsic on this target.
inline void mc_pause() {}

// Load eight bf16 values as one 16-byte vector.
inline sycl::vec<std::uint32_t, 4> load_raw8(const std::uint16_t *p) {
  return *reinterpret_cast<const sycl::vec<std::uint32_t, 4> *>(p);
}

// Unpack eight bf16 values to fp32.
inline void decode8(sycl::vec<std::uint32_t, 4> v, float *o) {
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const std::uint32_t word = v[i];
    o[2 * i] = mc_bf16_to_f32(static_cast<std::uint16_t>(word));
    o[2 * i + 1] = mc_bf16_to_f32(static_cast<std::uint16_t>(word >> 16));
  }
}

// Pack eight fp32 values and store them as bf16.
inline void store8(std::uint16_t *p, const float *a) {
  std::uint32_t packed[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const unsigned lo = mc_f32_to_bf16(a[2 * i]);
    const unsigned hi = mc_f32_to_bf16(a[2 * i + 1]);
    packed[i] = lo | (hi << 16);
  }
  *reinterpret_cast<sycl::vec<std::uint32_t, 4> *>(p) =
      sycl::vec<std::uint32_t, 4>(packed[0], packed[1], packed[2], packed[3]);
}

// One work-group publishes this epoch and waits until every rank has published
// this epoch or a later one. Ranks do not barrier between iterations, so a
// peer can already hold a newer epoch; equality would miss it and spin.
inline void peer_rendezvous(sycl::nd_item<1> item, unsigned long long *local_flags,
                            unsigned long long *const *peer_flags, int rank, int world,
                            unsigned long long epoch) {
  if (item.get_local_id(0) != 0)
    return;
  sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
  for (int peer = 0; peer < world; ++peer)
    mc_store_u64(peer_flags[peer] + rank, epoch);
  sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
  for (int peer = 0; peer < world; ++peer)
    while (mc_load_u64(local_flags + peer) < epoch)
      mc_pause();
  sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
}

// Weighted bf16 combine. The peer rendezvous has already completed.
template <int kTopk>
void combine_kernel(sycl::nd_item<1> item, int tokens, int hidden, const int *dest,
                    const int *slot, const std::uint16_t *const *stage, std::uint16_t *out) {
  const int nvec = hidden >> 3;
  const int thread = item.get_local_id(0);
  const int stride = static_cast<int>(item.get_local_range(0));
  const int blocks = item.get_group_range(0);
  for (int token = item.get_group(0); token < tokens; token += blocks) {
    for (int vec = thread; vec < nvec; vec += stride) {
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      const int *td = dest + static_cast<std::size_t>(token) * kTopk;
      const int *ts = slot + static_cast<std::size_t>(token) * kTopk;
      sycl::vec<std::uint32_t, 4> raw[kTopk];
      // A negative destination is not read. Dispatch keeps one expert per
      // destination rank, and a top-k larger than the pool leaves the rest empty.
#pragma unroll
      for (int k = 0; k < kTopk; ++k) {
        if (td[k] < 0)
          continue;
        const std::uint16_t *src =
            stage[td[k]] + (static_cast<std::size_t>(ts[k]) * hidden +
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
inline void fill_stage(sycl::nd_item<1> item, std::uint16_t *stage, int rank, int slots,
                       int hidden) {
  const std::size_t n = static_cast<std::size_t>(slots) * hidden;
  for (std::size_t i = item.get_global_id(0); i < n; i += item.get_global_range(0)) {
    const int local_slot = static_cast<int>(i / hidden);
    const int h = static_cast<int>(i - static_cast<std::size_t>(local_slot) * hidden);
    stage[i] = mc_f32_to_bf16(stage_unit(rank, local_slot, h));
  }
}
