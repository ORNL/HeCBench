// One launch per stage. A kernel return joins the whole grid, which is
// the grid barrier sppark writes as cooperative_groups::this_grid().sync()
// inside a single cudaLaunchCooperativeKernel. Stages:
//   1. breakdown  - signed window digits of every scalar
//   2. histogram  - how many points land in each (window, digit) bucket
//   3. scan       - exclusive prefix of those counts
//   4. scatter    - pack point indices, sign in bit 31, into bucket order
//   5. bucket_sum - one work-group sums each bucket
//   6. integrate  - total = 1*B[1] + 2*B[2] + ... + max_digit*B[max_digit]
//   7. fold       - shift each window by wbits and add the next

#pragma once

#include "msm.h"

__global__ void breakdown(int32_t* __restrict__ digits, const scalar_t* __restrict__ scalars,
                          uint32_t n, int wbits, int nwins) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int32_t d[kMaxWins];
  scalar_windows(scalars[i], d, wbits, nwins);
  for (int w = 0; w < nwins; w++) digits[(size_t)w * n + i] = d[w];
}

__global__ void histogram(const int32_t* __restrict__ digits, uint32_t* __restrict__ counts,
                          uint32_t n, int nwins, int nbins) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  for (int w = 0; w < nwins; w++) {
    int dig = digits[(size_t)w * n + i];
    if (dig == 0) continue;
    uint32_t k = dig < 0 ? -dig : dig;
    atomicAdd(counts + w * nbins + k, 1u);
  }
}

__global__ void exclusive_scan(const uint32_t* __restrict__ in, uint32_t* __restrict__ out,
                               int n) {
  uint32_t sum = 0;
  for (int i = 0; i < n; i++) {
    out[i] = sum;
    sum += in[i];
  }
}

__global__ void scatter(const int32_t* __restrict__ digits, uint32_t* __restrict__ cursor,
                        uint32_t* __restrict__ indices, uint32_t n, int nwins, int nbins) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  for (int w = 0; w < nwins; w++) {
    int dig = digits[(size_t)w * n + i];
    if (dig == 0) continue;
    uint32_t k = dig < 0 ? -dig : dig;
    uint32_t packed = i;
    if (dig < 0) packed |= 0x80000000u;
    uint32_t pos = atomicAdd(cursor + w * nbins + k, 1u);
    indices[pos] = packed;
  }
}

__global__ void bucket_sum(const uint32_t* __restrict__ indices,
                           const uint32_t* __restrict__ offsets, const uint32_t* __restrict__ counts,
                           const point_t* __restrict__ points, jac_t* __restrict__ buckets,
                           int nbins) {
  const int bucket = blockIdx.x + 1;
  const int win = blockIdx.y;
  const int slot = win * nbins + bucket;
  const uint32_t start = offsets[slot];
  const uint32_t len = counts[slot];

  jac_t acc = jinf();
  for (uint32_t j = threadIdx.x; j < len; j += blockDim.x) {
    uint32_t packed = indices[start + j];
    point_t p = points[packed & 0x7fffffffu];
    if (packed & 0x80000000u) p.y = fneg(p.y);
    acc = jadd(acc, jfrom(p));
  }

  __shared__ jac_t sm[kBlock];
  sm[threadIdx.x] = acc;
  __syncthreads();
  for (int off = kBlock >> 1; off > 0; off >>= 1) {
    if (threadIdx.x < off) sm[threadIdx.x] = jadd(sm[threadIdx.x], sm[threadIdx.x + off]);
    __syncthreads();
  }
  if (threadIdx.x == 0) buckets[slot] = sm[0];
}

__global__ void integrate(const jac_t* __restrict__ buckets, jac_t* __restrict__ wins,
                           int nbins, int max_digit) {
  int w = threadIdx.x;
  jac_t acc = jinf();
  jac_t sum = jinf();
  for (int k = max_digit; k >= 1; --k) {
    acc = jadd(acc, buckets[w * nbins + k]);
    sum = jadd(sum, acc);
  }
  wins[w] = sum;
}

__global__ void fold(const jac_t* __restrict__ wins, jac_t* __restrict__ out, int nwins,
                     int wbits) {
  jac_t r = wins[nwins - 1];
  for (int w = nwins - 2; w >= 0; --w) {
    for (int i = 0; i < wbits; i++) r = jdbl(r);
    r = jadd(r, wins[w]);
  }
  *out = r;
}
