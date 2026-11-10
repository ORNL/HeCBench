// One launch per stage. A kernel return joins the whole grid, which is
// the grid barrier sppark writes as cooperative_groups::this_grid().sync()
// inside a single cudaLaunchCooperativeKernel. Stages:
//   1. breakdown  - signed window digits of every scalar
//   2. histogram  - how many points land in each (window, digit) bucket
//   3. scan       - exclusive prefix of those counts and of the number of
//                   kSeg-point segments each bucket splits into
//   4. scatter    - pack point indices, sign in bit 31, into bucket order
//   5. bucket_sum - one thread sums each segment, so threads stay balanced
//                   when a narrow top window leaves a few very full buckets
//   6. merge      - add the segment sums of each bucket, one thread per
//                   bucket, or one work-group for buckets past kMerge segments
//   7. integrate  - one work-group per window forms
//                   1*B[1] + 2*B[2] + ... + max_digit*B[max_digit]
//                   from per-thread runs over contiguous buckets
// The host then folds the windows (fold_windows), as sppark's collect().
// Derived from sppark (Supranational LLC), Apache-2.0.

#pragma once

#include "msm.h"

static constexpr int kScan = 1024;
static constexpr int kSeg = 32;
static constexpr int kMerge = 64;

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

__device__ uint32_t block_exclusive(uint32_t local, uint32_t* sm) {
  const int tid = threadIdx.x;
  __syncthreads();
  sm[tid] = local;
  __syncthreads();
  for (int off = 1; off < kScan; off <<= 1) {
    uint32_t v = tid >= off ? sm[tid - off] : 0u;
    __syncthreads();
    sm[tid] += v;
    __syncthreads();
  }
  return sm[tid] - local;
}

__global__ void exclusive_scan(const uint32_t* __restrict__ counts, uint32_t* __restrict__ offsets,
                               uint32_t* __restrict__ cursor, uint32_t* __restrict__ seg_off,
                               uint32_t* __restrict__ nsegs, int n) {
  __shared__ uint32_t sm[kScan];
  const int tid = threadIdx.x;
  const int len = (n + kScan - 1) / kScan;
  const int lo = tid * len;
  const int hi = lo + len < n ? lo + len : n;
  uint32_t pts = 0, segs = 0;
  for (int i = lo; i < hi; i++) {
    pts += counts[i];
    segs += (counts[i] + kSeg - 1) / kSeg;
  }
  uint32_t p = block_exclusive(pts, sm);
  uint32_t q = block_exclusive(segs, sm);
  for (int i = lo; i < hi; i++) {
    offsets[i] = p;
    cursor[i] = p;
    seg_off[i] = q;
    p += counts[i];
    q += (counts[i] + kSeg - 1) / kSeg;
  }
  if (tid == kScan - 1) *nsegs = q;
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

// The bucket of a segment is the last one whose first segment is not past it.
__global__ void bucket_sum(const uint32_t* __restrict__ indices,
                           const uint32_t* __restrict__ offsets, const uint32_t* __restrict__ counts,
                           const uint32_t* __restrict__ seg_off, const uint32_t* __restrict__ nsegs,
                           const point_t* __restrict__ points, jac_t* __restrict__ partials,
                           int nslots) {
  const uint32_t seg = blockIdx.x * blockDim.x + threadIdx.x;
  if (seg >= *nsegs) return;
  int lo = 0, hi = nslots - 1;
  while (lo < hi) {
    int mid = (lo + hi + 1) >> 1;
    if (seg_off[mid] <= seg) lo = mid;
    else hi = mid - 1;
  }
  const uint32_t first = offsets[lo] + (seg - seg_off[lo]) * kSeg;
  const uint32_t end = offsets[lo] + counts[lo];
  const uint32_t last = first + kSeg < end ? first + kSeg : end;

  jac_t acc = jinf();
  for (uint32_t j = first; j < last; j++) {
    uint32_t packed = indices[j];
    point_t p = points[packed & 0x7fffffffu];
    if (packed & 0x80000000u) p.y = fneg(p.y);
    acc = jadd_mixed(acc, p);
  }
  partials[seg] = acc;
}

__global__ void bucket_merge(const jac_t* __restrict__ partials, const uint32_t* __restrict__ counts,
                             const uint32_t* __restrict__ seg_off, jac_t* __restrict__ buckets,
                             int nslots) {
  const int slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= nslots) return;
  const uint32_t nseg = (counts[slot] + kSeg - 1) / kSeg;
  if (nseg > kMerge) return;
  jac_t acc = jinf();
  for (uint32_t j = 0; j < nseg; j++) acc = jadd(acc, partials[seg_off[slot] + j]);
  buckets[slot] = acc;
}

__global__ void bucket_merge_wide(const jac_t* __restrict__ partials,
                                  const uint32_t* __restrict__ counts,
                                  const uint32_t* __restrict__ seg_off, jac_t* __restrict__ buckets) {
  const int slot = blockIdx.x;
  const int tid = threadIdx.x;
  const uint32_t nseg = (counts[slot] + kSeg - 1) / kSeg;
  if (nseg <= kMerge) return;
  jac_t acc = jinf();
  for (uint32_t j = tid; j < nseg; j += kBlock) acc = jadd(acc, partials[seg_off[slot] + j]);

  __shared__ jac_t sm[kBlock];
  sm[tid] = acc;
  __syncthreads();
  for (int off = kBlock >> 1; off > 0; off >>= 1) {
    if (tid < off) sm[tid] = jadd(sm[tid], sm[tid + off]);
    __syncthreads();
  }
  if (tid == 0) buckets[slot] = sm[0];
}

// Thread t owns buckets lo..hi. Its running sum weights them 1..(hi-lo+1),
// so it adds (lo-1) times their plain sum to get the true weights.
__global__ void integrate(const jac_t* __restrict__ buckets, jac_t* __restrict__ wins,
                          int nbins, int max_digit) {
  const int w = blockIdx.x;
  const int tid = threadIdx.x;
  const int len = (max_digit + kBlock - 1) / kBlock;
  const int lo = tid * len + 1;
  const int hi = lo + len - 1 < max_digit ? lo + len - 1 : max_digit;
  jac_t acc = jinf();
  jac_t sum = jinf();
  for (int k = hi; k >= lo; --k) {
    acc = jadd(acc, buckets[w * nbins + k]);
    sum = jadd(sum, acc);
  }
  if (lo > 1) sum = jadd(sum, jmul(acc, lo - 1));

  __shared__ jac_t sm[kBlock];
  sm[tid] = sum;
  __syncthreads();
  for (int off = kBlock >> 1; off > 0; off >>= 1) {
    if (tid < off) sm[tid] = jadd(sm[tid], sm[tid + off]);
    __syncthreads();
  }
  if (tid == 0) wins[w] = sm[0];
}
