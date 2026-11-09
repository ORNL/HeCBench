// One launch per stage. Separate submissions join the grid between stages,
// the same role as a root-group barrier. Stages:
//   1. breakdown  - signed window digits of every scalar
//   2. histogram  - how many points land in each (window, digit) bucket
//   3. scan       - exclusive prefix of those counts
//   4. scatter    - pack point indices, sign in bit 31, into bucket order
//   5. bucket_sum - one work-group sums each bucket
//   6. integrate  - total = 1*B[1] + 2*B[2] + ... + max_digit*B[max_digit]
//   7. fold       - shift each window by wbits and add the next

#pragma once

#include <sycl/sycl.hpp>

#include "msm.h"

inline void breakdown(int32_t* __restrict__ digits, const scalar_t* __restrict__ scalars, uint32_t n,
                      int wbits, int nwins, sycl::nd_item<1> it) {
  uint32_t i = it.get_global_id(0);
  if (i >= n) return;
  int32_t d[kMaxWins];
  scalar_windows(scalars[i], d, wbits, nwins);
  for (int w = 0; w < nwins; w++) digits[(size_t)w * n + i] = d[w];
}

inline void histogram(const int32_t* __restrict__ digits, uint32_t* __restrict__ counts, uint32_t n,
                      int nwins, int nbins, sycl::nd_item<1> it) {
  uint32_t i = it.get_global_id(0);
  if (i >= n) return;
  for (int w = 0; w < nwins; w++) {
    int dig = digits[(size_t)w * n + i];
    if (dig == 0) continue;
    uint32_t k = dig < 0 ? -dig : dig;
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        slot(counts[w * nbins + k]);
    slot.fetch_add(1u);
  }
}

inline void exclusive_scan(const uint32_t* __restrict__ in, uint32_t* __restrict__ out, int n) {
  uint32_t sum = 0;
  for (int i = 0; i < n; i++) {
    out[i] = sum;
    sum += in[i];
  }
}

inline void scatter(const int32_t* __restrict__ digits, uint32_t* __restrict__ cursor,
                    uint32_t* __restrict__ indices, uint32_t n, int nwins, int nbins,
                    sycl::nd_item<1> it) {
  uint32_t i = it.get_global_id(0);
  if (i >= n) return;
  for (int w = 0; w < nwins; w++) {
    int dig = digits[(size_t)w * n + i];
    if (dig == 0) continue;
    uint32_t k = dig < 0 ? -dig : dig;
    uint32_t packed = i;
    if (dig < 0) packed |= 0x80000000u;
    sycl::atomic_ref<uint32_t, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        slot(cursor[w * nbins + k]);
    uint32_t pos = slot.fetch_add(1u);
    indices[pos] = packed;
  }
}

inline void bucket_sum(const uint32_t* __restrict__ indices, const uint32_t* __restrict__ offsets,
                       const uint32_t* __restrict__ counts, const point_t* __restrict__ points,
                       jac_t* __restrict__ buckets, jac_t* __restrict__ sm, int nbins,
                       sycl::nd_item<2> it) {
  const int bucket = it.get_group(1) + 1;
  const int win = it.get_group(0);
  const int slot = win * nbins + bucket;
  const uint32_t start = offsets[slot];
  const uint32_t len = counts[slot];
  const int tid = it.get_local_id(1);

  jac_t acc = jinf();
  for (uint32_t j = tid; j < len; j += it.get_local_range(1)) {
    uint32_t packed = indices[start + j];
    point_t p = points[packed & 0x7fffffffu];
    if (packed & 0x80000000u) p.y = fneg(p.y);
    acc = jadd(acc, jfrom(p));
  }
  sm[tid] = acc;
  sycl::group_barrier(it.get_group());
  for (int off = kBlock >> 1; off > 0; off >>= 1) {
    if (tid < off) sm[tid] = jadd(sm[tid], sm[tid + off]);
    sycl::group_barrier(it.get_group());
  }
  if (tid == 0) buckets[slot] = sm[0];
}

inline void integrate(const jac_t* __restrict__ buckets, jac_t* __restrict__ wins, int nbins,
                      int max_digit, sycl::nd_item<1> it) {
  int w = it.get_local_id(0);
  jac_t acc = jinf();
  jac_t sum = jinf();
  for (int k = max_digit; k >= 1; --k) {
    acc = jadd(acc, buckets[w * nbins + k]);
    sum = jadd(sum, acc);
  }
  wins[w] = sum;
}

inline void fold(const jac_t* __restrict__ wins, jac_t* __restrict__ out, int nwins, int wbits) {
  jac_t r = wins[nwins - 1];
  for (int w = nwins - 2; w >= 0; --w) {
    for (int i = 0; i < wbits; i++) r = jdbl(r);
    r = jadd(r, wins[w]);
  }
  *out = r;
}
