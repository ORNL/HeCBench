#pragma once

#include <stddef.h>

// Host-only launch heuristics shared by the CUDA, HIP, SYCL, and OpenMP
// variants. These choices are portable, not universally optimal: subgroup
// width, register allocation, cache behavior, and occupancy differ by GPU
// vendor and architecture. Benchmark representative workloads on each target
// and adjust the policy inputs or heuristics before treating them as tuned.

struct SSM2DeviceInfo {
  size_t compute_units;
  int max_threads_per_block;
  int subgroup_width;
};

struct SSM2LaunchPolicy {
  int state_bh;
  int state_bs;
  int out_bh;
  int state_splits;
  int scalar_state_threads;
  int scalar_out_threads;
};

// Pick a subgroup-aligned work-group size while keeping enough groups to expose
// roughly four waves of work per compute unit. This is used by scalar kernels,
// where changing the work-group size does not change the numerical algorithm.
inline int ssm2_pick_scalar_threads(size_t work, const SSM2DeviceInfo &device) {
  const int subgroup = device.subgroup_width > 0 ? device.subgroup_width : 32;
  int cap =
      device.max_threads_per_block > 0 ? device.max_threads_per_block : 256;
  if (cap > 256)
    cap = 256;
  cap -= cap % subgroup;
  if (cap < subgroup)
    cap = subgroup;

  int threads = cap;
  const size_t target_groups =
      (device.compute_units ? device.compute_units : 1) * 4;
  while (threads > subgroup &&
         (work + (size_t)threads - 1) / (size_t)threads < target_groups)
    threads /= 2;
  if (threads < subgroup)
    threads = subgroup;
  return threads;
}

// Select a vectorized P-by-N state tile. One thread owns a 4x4 fragment, so
// candidates must divide P/N exactly and produce a subgroup-aligned block.
// Prefer more independent blocks, then balanced tiles, then larger blocks.
inline void ssm2_pick_state_tile(int batches, int heads, int P, int N,
                                 const SSM2DeviceInfo &device, int *BH,
                                 int *BS) {
  const int options[] = {8, 16, 32, 64, 4};
  const int subgroup = device.subgroup_width > 0 ? device.subgroup_width : 32;
  int thread_cap =
      device.max_threads_per_block > 0 ? device.max_threads_per_block : 256;
  if (thread_cap > 256)
    thread_cap = 256;

  size_t best_blocks = 0;
  int best_balance = -1, best_threads = -1;
  int best_bh = 0, best_bs = 0;
  for (int bh : options) {
    if (bh > P || P % bh != 0)
      continue;
    for (int bs : options) {
      if (bs > N || N % bs != 0)
        continue;
      const int threads = (bh / 4) * (bs / 4);
      if (threads < subgroup || threads > thread_cap || threads % subgroup != 0)
        continue;
      const size_t blocks = (size_t)batches * heads * (P / bh) * (N / bs);
      const int balance = bh < bs ? bh : bs;
      if (blocks > best_blocks ||
          (blocks == best_blocks && balance > best_balance) ||
          (blocks == best_blocks && balance == best_balance &&
           threads > best_threads)) {
        best_blocks = blocks;
        best_balance = balance;
        best_threads = threads;
        best_bh = bh;
        best_bs = bs;
      }
    }
  }
  *BH = best_bh;
  *BS = best_bs;
}

// Split a state tile's time reduction when longer chunks or a small tile grid
// can profit from more resident warps. Each split must preserve enough time
// work to amortize storing and reducing its per-thread 4x4 partial.
inline int ssm2_pick_state_splits(int chunk, int tile_threads,
                                  size_t state_blocks,
                                  const SSM2DeviceInfo &device) {
  int thread_cap =
      device.max_threads_per_block > 0 ? device.max_threads_per_block : 256;
  if (thread_cap > 256)
    thread_cap = 256;
  const size_t compute_units = device.compute_units ? device.compute_units : 1;
  if (tile_threads > 0 && chunk % 4 == 0 && chunk / 4 >= 16 &&
      tile_threads * 4 <= thread_cap)
    return 4;
  if (tile_threads > 0 && chunk % 2 == 0 &&
      (chunk / 2 >= 32 || state_blocks < compute_units) &&
      tile_threads * 2 <= thread_cap)
    return 2;
  return 1;
}

// Select the largest exact P tile whose chunk-by-P grid of 4x4 fragments fits
// in one work-group.
inline int ssm2_pick_out_tile(int P, int chunk, const SSM2DeviceInfo &device) {
  const int options[] = {64, 32, 16, 8, 4};
  const int thread_cap =
      device.max_threads_per_block > 0 ? device.max_threads_per_block : 256;
  for (int bh : options) {
    const int threads = (chunk / 4) * (bh / 4);
    if (bh <= P && P % bh == 0 && threads > 0 && threads <= thread_cap)
      return bh;
  }
  return 0;
}

// Build one policy from the problem shape and device properties. CUDA, HIP,
// and SYCL consume the tiled GEMM fields when their resource checks admit the
// optimized path. Their scalar kernels use a fixed 256-thread group so the
// tiled state update can cover one 16x16 tile. OpenMP consumes the adaptive
// thread counts.
inline SSM2LaunchPolicy
ssm2_select_launch_policy(int B, int T, int H, int P, int N, int chunk,
                          const SSM2DeviceInfo &device) {
  SSM2LaunchPolicy policy{};
  ssm2_pick_state_tile(B, H, P, N, device, &policy.state_bh, &policy.state_bs);
  policy.out_bh = ssm2_pick_out_tile(P, chunk, device);
  const int tile_threads = policy.state_bh && policy.state_bs
                               ? (policy.state_bh / 4) * (policy.state_bs / 4)
                               : 0;
  const size_t state_blocks =
      tile_threads
          ? (size_t)B * H * (P / policy.state_bh) * (N / policy.state_bs)
          : 0;
  policy.state_splits =
      ssm2_pick_state_splits(chunk, tile_threads, state_blocks, device);
  policy.scalar_state_threads =
      ssm2_pick_scalar_threads((size_t)B * H * P * N, device);
  policy.scalar_out_threads =
      ssm2_pick_scalar_threads((size_t)B * T * H * P, device);
  return policy;
}
