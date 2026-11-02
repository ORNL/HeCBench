/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Adapted from Meta Superintelligence Labs Kernels (MSLK) index_shuffling:
 * https://github.com/meta-pytorch/MSLK/blob/main/csrc/moe/index_shuffling.cu
 *
 * Cooperative top-k + histogram + expert-order index placement for MoE routing
 * (MetaShuffling IndexShuffling). A second, two-kernel path does the same
 * work without the grid barrier: histogram, then placement.
 */

#pragma once

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>

// hipcc defines __HIPCC__
#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaStream_t hipStream_t
#define cudaDeviceProp hipDeviceProp_t
#define cudaMalloc hipMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetLastError hipGetLastError
#define cudaLaunchCooperativeKernel hipLaunchCooperativeKernel
#else
// nvcc keeps the CUDA runtime and the PTX atomics.
#include <cuda_runtime.h>
#endif

constexpr int kBlockSize = 256;

#define CHECK(call)                                                            \
  do {                                                                         \
    const cudaError_t err = (call);                                            \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "GPU error %s:%d '%s': %s\n", __FILE__, __LINE__,        \
              #call, cudaGetErrorString(err));                                 \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// Round a / b up to the next integer.
__host__ __device__ __forceinline__ constexpr int ceil_of_ratio(int a, int b) {
  return (a + b - 1) / b;
}

#if !defined(__HIPCC__) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
// Return the old value of a relaxed, device-wide atomic add.
__device__ __forceinline__ int atomic_add_relaxed(int *addr, int inc) {
  int val;
  asm volatile("atom.relaxed.gpu.global.add.s32 %0, [%1], %2;\n"
               : "=r"(val)
               : "l"(addr), "r"(inc));
  return val;
}

// Return the old value of an atomic add that releases earlier writes.
__device__ __forceinline__ int atomic_add_release(int *addr, int inc) {
  int val;
  asm volatile("atom.release.gpu.global.add.s32 %0, [%1], %2;\n"
               : "=r"(val)
               : "l"(addr), "r"(inc));
  return val;
}

// Load an int after preceding device writes are visible.
__device__ __forceinline__ int load_acquire(int *addr) {
  int val;
  asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n"
               : "=r"(val)
               : "l"(addr));
  return val;
}
#elif defined(__HIPCC__)
// Return the old value of an atomic add.
__device__ __forceinline__ int atomic_add_relaxed(int *addr, int inc) {
  return __hip_atomic_fetch_add(addr, inc, __ATOMIC_RELAXED,
                                __HIP_MEMORY_SCOPE_AGENT);
}

// Publish preceding device writes before adding to the grid check-in counter.
__device__ __forceinline__ int atomic_add_release(int *addr, int inc) {
  __threadfence();
  return __hip_atomic_fetch_add(addr, inc, __ATOMIC_RELEASE,
                                __HIP_MEMORY_SCOPE_AGENT);
}

// Atomically load the grid check-in counter with acquire ordering.
__device__ __forceinline__ int load_acquire(int *addr) {
  return __hip_atomic_load(addr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
}
#else
// Pre-Volta CUDA lacks the PTX acquire/release forms above. An atomic RMW
// avoids a conflicting plain load; fences supply the matching release/acquire
// ordering around it.
__device__ __forceinline__ int atomic_add_relaxed(int *addr, int inc) {
  return atomicAdd(addr, inc);
}

__device__ __forceinline__ int atomic_add_release(int *addr, int inc) {
  __threadfence();
  return atomicAdd(addr, inc);
}

__device__ __forceinline__ int load_acquire(int *addr) {
  const int val = atomicAdd(addr, 0);
  __threadfence();
  return val;
}
#endif

template <class DataType, class IndexType>
struct Params {
  const DataType *routing_scores;
  int stride_t;
  int stride_e;
  int expert_index_start;
  int expert_index_end;
  const IndexType *valid_token_count;
  int num_tokens;
  int num_tokens_per_cta;

  IndexType *buffered_expert_indices;
  IndexType *buffered_token_indices;

  IndexType *token_count_per_expert;
  IndexType *shuffled_expert_indices;
  IndexType *shuffled_token_indices;
};

// Exchange a register with the lane whose id differs by offset, inside a
// segment of Width lanes. Width is a power of two, so an offset below Width
// never leaves the segment. HIP skips the sync form: its mask check is a
// software loop, and every lane of the wave executes this together.
template <int Width, class T>
__device__ __forceinline__ T shfl_xor(T value, int offset) {
#if defined(__HIPCC__)
  return __shfl_xor(value, offset, Width);
#else
  return __shfl_xor_sync(0xffffffffu, value, offset, Width);
#endif
}

// Select one token's top-k experts. The calling warp is split into segments of
// Width = min(warp size, NumExperts) lanes, and every lane in a segment holds
// the same token. Lane `seg_lane` owns experts seg_lane, seg_lane + Width, ...
// so the segment's first load is a contiguous run of scores.
//
// Those scores are kept in registers and reused for every round. A round drops
// experts already chosen, then an XOR butterfly keeps the highest score and,
// on a tie, the smaller expert index. The butterfly result is broadcast to
// every lane of the segment, which the next round's exclusion test needs.
// Inactive lanes load nothing and still join the butterfly, so the warp stays
// convergent; their -infinity scores cannot win.
template <class DataType, int NumExperts, int TopK, int Width>
__device__ __forceinline__ void
select_topk_segment(const DataType *row, int stride_e, int seg_lane,
                    bool active, int *picked) {
  constexpr int kPerLane = NumExperts / Width;
  static_assert(NumExperts % Width == 0, "segment must tile the expert row");
  static_assert(TopK <= Width, "a segment lane writes each winner");

  DataType lane_scores[kPerLane];
#pragma unroll
  for (int i = 0; i < kPerLane; ++i)
    lane_scores[i] = static_cast<DataType>(-INFINITY);
  if (active) {
#pragma unroll
    for (int i = 0; i < kPerLane; ++i)
      lane_scores[i] = row[(seg_lane + i * Width) * stride_e];
  }

#pragma unroll
  for (int k = 0; k < TopK; ++k) {
    DataType best_score = static_cast<DataType>(-INFINITY);
    int best_expert = INT_MAX;
#pragma unroll
    for (int i = 0; i < kPerLane; ++i) {
      const int expert = seg_lane + i * Width;
      bool taken = false;
#pragma unroll
      for (int p = 0; p < k; ++p)
        taken |= picked[p] == expert;
      const DataType score = lane_scores[i];
      if (!taken && (score > best_score ||
                     (score == best_score && expert < best_expert))) {
        best_score = score;
        best_expert = expert;
      }
    }
#pragma unroll
    for (int offset = Width / 2; offset > 0; offset >>= 1) {
      const DataType other_score = shfl_xor<Width>(best_score, offset);
      const int other_expert = shfl_xor<Width>(best_expert, offset);
      if (other_score > best_score ||
          (other_score == best_score && other_expert < best_expert)) {
        best_score = other_score;
        best_expert = other_expert;
      }
    }
    picked[k] = best_expert;
  }
}

// Fused IndexShuffling kernel.
//
// One grid does four things for a token slice:
//   1. Reduce each token's expert scores to its top-k experts.
//   2. Atomically count how many selected tokens land on each expert.
//   3. Wait until every block has published its counts.
//   4. Write token indices so each expert occupies one contiguous output run.
//
// Step 1 gives each token a segment of min(warpSize, NumExperts) lanes.
// Sixteen experts fill two segments of a 32-lane warp and four segments of a
// 64-lane wave. The lanes read adjacent scores instead of replicating every
// score into shared memory TopK times and merging it with a block-wide
// tournament.
//
// The output order inside an expert is the order of the atomic counter updates,
// so it is not deterministic. The verifier checks counts, run boundaries, and
// the token permutation rather than that order.
//
// The wait in step 3 is a software grid barrier. A cooperative launch is what
// keeps every block resident; otherwise a spinning block can prevent the
// missing blocks from ever incrementing the counter.
template <class DataType, class IndexType, int NumExperts, int TopK, int WarpSize>
__device__ __forceinline__ void index_shuffling_warp(
    Params<DataType, IndexType> params, IndexType *token_count_cumsums) {
  constexpr int kWidth = (NumExperts < WarpSize) ? NumExperts : WarpSize;
  constexpr int kSegments = WarpSize / kWidth;
  constexpr int kTokensPerStep = kBlockSize / kWidth;

  const auto tidx = threadIdx.x;
  const auto bidx = blockIdx.x;
  const int lane = tidx & (WarpSize - 1);
  const int seg_lane = lane & (kWidth - 1);
  const int seg = lane / kWidth;
  const int warp_in_block = tidx / WarpSize;

  const int num_total_tokens = params.num_tokens;
  const int token_index_start = bidx * params.num_tokens_per_cta;
  // An empty tail block adds nothing and still has to reach the barrier.
  const int range_begin = min(token_index_start, num_total_tokens);
  const int range_end =
      min(token_index_start + params.num_tokens_per_cta, num_total_tokens);

  const int num_valid_tokens =
      params.valid_token_count ? *params.valid_token_count : num_total_tokens;

  const int expert_index_start = params.expert_index_start;
  const int expert_index_end = params.expert_index_end;
  const int stride_t = params.stride_t;
  const int stride_e = params.stride_e;

  for (int token_base = range_begin; token_base < range_end;
       token_base += kTokensPerStep) {
    const int token = token_base + warp_in_block * kSegments + seg;
    // Every lane of a segment shares `token`, so this is uniform inside the
    // segment. Lanes past the block's range stay for the butterfly.
    const bool active = token < range_end && token < num_valid_tokens;
    const DataType *row =
        active ? params.routing_scores + static_cast<long long>(token) * stride_t
               : nullptr;

    int picked[TopK];
#pragma unroll
    for (int k = 0; k < TopK; ++k)
      picked[k] = -1;
    select_topk_segment<DataType, NumExperts, TopK, kWidth>(
        row, stride_e, seg_lane, active, picked);

    // Segment lane k saves and counts the k-th winner. The rank is this
    // token's position among tokens routed to that expert; the final output
    // index also needs the expert's prefix, which is not known until every
    // block has published its counts.
    if (active) {
#pragma unroll
      for (int k = 0; k < TopK; ++k) {
        if (seg_lane == k) {
          const int expert = picked[k];
          params.buffered_expert_indices[token * TopK + k] = expert;
          if (expert >= expert_index_start && expert < expert_index_end) {
            const int rank = atomic_add_relaxed(
                &params.token_count_per_expert[expert], 1);
            params.buffered_token_indices[token * TopK + k] = rank;
          }
        }
      }
    }
  }

  // counts[NumExperts] is the grid check-in, not an expert bin. Make this
  // block's histogram atomics visible to thread 0, which then publishes the
  // block's token count with release ordering and acquire-spins until every
  // block has checked in. The release makes those histogram updates visible
  // to any block that observes the final total.
  __syncthreads();
  if (tidx == 0) {
    int *processed_tokens_addr = &params.token_count_per_expert[NumExperts];
    const int inc = range_end - range_begin;
    atomic_add_release(processed_tokens_addr, inc);
    int processed_tokens;
    do {
      processed_tokens = load_acquire(processed_tokens_addr);
    } while (processed_tokens != num_total_tokens);
  }
  // Publish the completed wait to the rest of the block before anyone reads
  // the histogram.
  __syncthreads();

  // Exclusive scan of the expert counts. After the scan, entry i holds the
  // total for experts 0..i, so expert i begins at entry i - 1 (or at 0).
  for (int i = tidx; i < NumExperts; i += kBlockSize)
    token_count_cumsums[i] = params.token_count_per_expert[i];
  __syncthreads();

  if (tidx == 0) {
#pragma unroll
    for (int i = 1; i < NumExperts; ++i)
      token_count_cumsums[i] += token_count_cumsums[i - 1];
  }
  __syncthreads();

  // Return the number of selected tokens assigned to experts before index.
  auto prefix = [&](int index) {
    return index == 0 ? 0 : token_count_cumsums[index - 1];
  };

  const int token_count_cumsum_start = prefix(expert_index_start);
  const int token_count_cumsum_end = prefix(expert_index_end);
  const int num_selected_tokens =
      token_count_cumsum_end - token_count_cumsum_start;

  // Place each selected token at prefix[expert] + rank. Experts outside
  // [expert_index_start, expert_index_end) were counted but are not written.
  // The stored expert id is relative to expert_index_start.
  for (int token_base = range_begin; token_base < range_end;
       token_base += kBlockSize) {
    const int token = token_base + tidx;
    if (token < range_end && token < num_valid_tokens) {
#pragma unroll
      for (int k = 0; k < TopK; ++k) {
        const int expert = params.buffered_expert_indices[token * TopK + k];
        if (expert >= expert_index_start && expert < expert_index_end) {
          const int rank = params.buffered_token_indices[token * TopK + k];
          const int output =
              prefix(expert) - token_count_cumsum_start + rank;
          params.shuffled_expert_indices[output] = expert - expert_index_start;
          params.shuffled_token_indices[output] = token;
        }
      }
    }
  }

  // counts[NumExperts + 1] reports how many tokens this expert range owns.
  if (tidx == 0 && bidx == 0)
    params.token_count_per_expert[NumExperts + 1] = num_selected_tokens;
}

// warpSize is the same for every thread, so exactly one specialization runs
// and the block-wide barriers inside it stay convergent.
template <class DataType, class IndexType, int NumExperts, int TopK>
__global__ void index_shuffling_kernel(Params<DataType, IndexType> params) {
  // Shared copy of the expert counts for the exclusive scan below.
  __shared__ IndexType token_count_cumsums[NumExperts];
  if (warpSize == 64) {
    index_shuffling_warp<DataType, IndexType, NumExperts, TopK, 64>(
        params, token_count_cumsums);
  } else {
    index_shuffling_warp<DataType, IndexType, NumExperts, TopK, 32>(
        params, token_count_cumsums);
  }
}

// Launch one cooperative grid covering the tokens. Cap it at the number of
// SMs so the grid barrier's blocks can stay resident.
template <int NumExperts, int TopK>
void launch_index_shuffling(const float *d_scores, int num_tokens,
                            int *d_token_count_per_expert,
                            int *d_shuffled_expert_indices,
                            int *d_shuffled_token_indices,
                            int *d_buffered_expert_indices,
                            int *d_buffered_token_indices, int num_sms,
                            int warp_size, cudaStream_t stream) {
  using DataType = float;
  using IndexType = int;

  // Match the kernel: a 64-lane device uses warpSize 64, everything else 32.
  const int logical_warp = warp_size == 64 ? 64 : 32;
  const int width = (NumExperts < logical_warp) ? NumExperts : logical_warp;
  const int tokens_per_step = kBlockSize / width;
  const int num_ctas = std::min(ceil_of_ratio(num_tokens, tokens_per_step),
                                std::max(num_sms, 1));
  const int num_tokens_per_cta = ceil_of_ratio(num_tokens, num_ctas);

  Params<DataType, IndexType> params;
  params.routing_scores = d_scores;
  params.stride_t = NumExperts;
  params.stride_e = 1;
  params.expert_index_start = 0;
  params.expert_index_end = NumExperts;
  params.valid_token_count = nullptr;
  params.num_tokens = num_tokens;
  params.num_tokens_per_cta = num_tokens_per_cta;
  params.buffered_expert_indices = d_buffered_expert_indices;
  params.buffered_token_indices = d_buffered_token_indices;
  params.token_count_per_expert = d_token_count_per_expert;
  params.shuffled_expert_indices = d_shuffled_expert_indices;
  params.shuffled_token_indices = d_shuffled_token_indices;

  void *kernel =
      (void *)index_shuffling_kernel<DataType, IndexType, NumExperts, TopK>;
  dim3 grids(num_ctas);
  dim3 blocks(kBlockSize);
  void *args[] = {(void *)&params};

  // Cooperative launch guarantees the grid barrier can make progress. A
  // normal launch is invalid here: a spinning resident block could prevent an
  // unscheduled block from reaching the check-in, so reject unsupported grids.
  cudaError_t launch_status = cudaLaunchCooperativeKernel(
      kernel, grids, blocks, args, 0, stream);
  if (launch_status != cudaSuccess) {
    fprintf(stderr, "GPU error %s:%d cudaLaunchCooperativeKernel: %s\n",
            __FILE__, __LINE__, cudaGetErrorString(launch_status));
    exit(EXIT_FAILURE);
  }
}

// Check the supported shape and launch its compiled specialization.
inline void dispatch_index_shuffling(const float *d_scores, int num_tokens,
                                     int num_experts, int top_k,
                                     int *d_token_count_per_expert,
                                     int *d_shuffled_expert_indices,
                                     int *d_shuffled_token_indices,
                                     int *d_buffered_expert_indices,
                                     int *d_buffered_token_indices, int num_sms,
                                     int warp_size, cudaStream_t stream) {
  if (!(top_k == 1 || top_k == 2 || top_k == 4)) {
    fprintf(stderr, "Unsupported top_k=%d (supported: 1, 2, 4)\n", top_k);
    exit(EXIT_FAILURE);
  }
  if (!(num_experts == 16 || num_experts == 32 || num_experts == 128)) {
    fprintf(stderr,
            "Unsupported num_experts=%d (supported: 16, 32, 128)\n",
            num_experts);
    exit(EXIT_FAILURE);
  }

#define LAUNCH(E, K)                                                           \
  launch_index_shuffling<E, K>(                                                \
      d_scores, num_tokens, d_token_count_per_expert,                          \
      d_shuffled_expert_indices, d_shuffled_token_indices,                     \
      d_buffered_expert_indices, d_buffered_token_indices, num_sms,         \
      warp_size, stream)

  if (num_experts == 16 && top_k == 1)
    LAUNCH(16, 1);
  else if (num_experts == 16 && top_k == 2)
    LAUNCH(16, 2);
  else if (num_experts == 16 && top_k == 4)
    LAUNCH(16, 4);
  else if (num_experts == 32 && top_k == 1)
    LAUNCH(32, 1);
  else if (num_experts == 32 && top_k == 2)
    LAUNCH(32, 2);
  else if (num_experts == 32 && top_k == 4)
    LAUNCH(32, 4);
  else if (num_experts == 128 && top_k == 1)
    LAUNCH(128, 1);
  else if (num_experts == 128 && top_k == 2)
    LAUNCH(128, 2);
  else
    LAUNCH(128, 4);
#undef LAUNCH
}

// Histogram half of the two-kernel path. One segment selects one token, the
// same way as the fused kernel, then lane k stores that winner and counts it.
// The atomic's old value is not a rank: placement runs in a later kernel,
// after every block has finished this one.
template <class DataType, int NumExperts, int TopK, int WarpSize>
__device__ __forceinline__ void index_shuffling_histogram_warp(
    const DataType *routing_scores, int num_tokens, int *counts, int *chosen) {
  constexpr int kWidth = (NumExperts < WarpSize) ? NumExperts : WarpSize;
  constexpr int kSegments = WarpSize / kWidth;
  constexpr int kTokensPerStep = kBlockSize / kWidth;

  const int tidx = threadIdx.x;
  const int lane = tidx & (WarpSize - 1);
  const int seg_lane = lane & (kWidth - 1);
  const int seg = lane / kWidth;
  const int warp_in_block = tidx / WarpSize;

  const int token =
      blockIdx.x * kTokensPerStep + warp_in_block * kSegments + seg;
  // Lanes past the last token still join the butterfly.
  const bool active = token < num_tokens;
  const DataType *row =
      active ? routing_scores + static_cast<long long>(token) * NumExperts
             : nullptr;

  int picked[TopK];
#pragma unroll
  for (int k = 0; k < TopK; ++k)
    picked[k] = -1;
  select_topk_segment<DataType, NumExperts, TopK, kWidth>(
      row, /*stride_e=*/1, seg_lane, active, picked);

  if (active) {
#pragma unroll
    for (int k = 0; k < TopK; ++k) {
      if (seg_lane == k) {
        const int expert = picked[k];
        chosen[token * TopK + k] = expert;
        atomic_add_relaxed(&counts[expert], 1);
      }
    }
  }
}

// warpSize is the same for every thread, so exactly one specialization runs
// and the segment butterfly stays convergent.
template <int NumExperts, int TopK>
__global__ void index_shuffling_histogram_kernel(const float *routing_scores,
                                                 int num_tokens, int *counts,
                                                 int *chosen) {
  if (warpSize == 64) {
    index_shuffling_histogram_warp<float, NumExperts, TopK, 64>(
        routing_scores, num_tokens, counts, chosen);
  } else {
    index_shuffling_histogram_warp<float, NumExperts, TopK, 32>(
        routing_scores, num_tokens, counts, chosen);
  }
}

// Placement half of the two-kernel path. Each block scans the finished
// histogram into exclusive offsets, then one thread per selected slot claims
// the next slot of its expert. positions[e] starts at zero, so the returned
// values are 0 .. counts[e] - 1 and the writes form one contiguous run.
template <int NumExperts, int TopK>
__global__ void index_shuffling_place_kernel(const int *chosen, int *counts,
                                             int *positions,
                                             int *shuffled_experts,
                                             int *shuffled_tokens,
                                             int num_tokens) {
  __shared__ int offsets[NumExperts];
  const int tidx = threadIdx.x;
  if (tidx < NumExperts)
    offsets[tidx] = counts[tidx];
  __syncthreads();

  if (tidx == 0) {
    int sum = 0;
#pragma unroll
    for (int i = 0; i < NumExperts; ++i) {
      const int count = offsets[i];
      offsets[i] = sum;
      sum += count;
    }
    // counts[NumExperts + 1] reports how many tokens were selected.
    if (blockIdx.x == 0)
      counts[NumExperts + 1] = sum;
  }
  __syncthreads();

  const int total = num_tokens * TopK;
  const int flat = blockIdx.x * kBlockSize + tidx;
  if (flat >= total)
    return;
  const int token = flat / TopK;
  const int expert = chosen[flat];
  const int rank = atomic_add_relaxed(&positions[expert], 1);
  const int output = offsets[expert] + rank;
  shuffled_experts[output] = expert;
  shuffled_tokens[output] = token;
}

// Launch the histogram grid, then the placement grid, on one stream. No
// cooperative launch: finishing the first kernel publishes the counts.
template <int NumExperts, int TopK>
void launch_index_shuffling_split(const float *d_scores, int num_tokens,
                                  int *d_counts, int *d_positions, int *d_chosen,
                                  int *d_shuffled_expert_indices,
                                  int *d_shuffled_token_indices, int warp_size,
                                  cudaStream_t stream) {
  const int logical_warp = warp_size == 64 ? 64 : 32;
  const int width = (NumExperts < logical_warp) ? NumExperts : logical_warp;
  const int tokens_per_block = kBlockSize / width;
  const int hist_blocks = ceil_of_ratio(num_tokens, tokens_per_block);
  const int total = num_tokens * TopK;
  const int place_blocks = ceil_of_ratio(total, kBlockSize);

  index_shuffling_histogram_kernel<NumExperts, TopK>
      <<<hist_blocks, kBlockSize, 0, stream>>>(d_scores, num_tokens, d_counts,
                                               d_chosen);
  CHECK(cudaGetLastError());
  index_shuffling_place_kernel<NumExperts, TopK>
      <<<place_blocks, kBlockSize, 0, stream>>>(
          d_chosen, d_counts, d_positions, d_shuffled_expert_indices,
          d_shuffled_token_indices, num_tokens);
  CHECK(cudaGetLastError());
}

// Check the supported shape and launch the two-kernel specialization.
inline void dispatch_index_shuffling_split(
    const float *d_scores, int num_tokens, int num_experts, int top_k,
    int *d_counts, int *d_positions, int *d_chosen,
    int *d_shuffled_expert_indices, int *d_shuffled_token_indices,
    int warp_size, cudaStream_t stream) {
  if (!(top_k == 1 || top_k == 2 || top_k == 4)) {
    fprintf(stderr, "Unsupported top_k=%d (supported: 1, 2, 4)\n", top_k);
    exit(EXIT_FAILURE);
  }
  if (!(num_experts == 16 || num_experts == 32 || num_experts == 128)) {
    fprintf(stderr, "Unsupported num_experts=%d (supported: 16, 32, 128)\n",
            num_experts);
    exit(EXIT_FAILURE);
  }

#define LAUNCH_SPLIT(E, K)                                                     \
  launch_index_shuffling_split<E, K>(                                         \
      d_scores, num_tokens, d_counts, d_positions, d_chosen,                   \
      d_shuffled_expert_indices, d_shuffled_token_indices, warp_size, stream)

  if (num_experts == 16 && top_k == 1)
    LAUNCH_SPLIT(16, 1);
  else if (num_experts == 16 && top_k == 2)
    LAUNCH_SPLIT(16, 2);
  else if (num_experts == 16 && top_k == 4)
    LAUNCH_SPLIT(16, 4);
  else if (num_experts == 32 && top_k == 1)
    LAUNCH_SPLIT(32, 1);
  else if (num_experts == 32 && top_k == 2)
    LAUNCH_SPLIT(32, 2);
  else if (num_experts == 32 && top_k == 4)
    LAUNCH_SPLIT(32, 4);
  else if (num_experts == 128 && top_k == 1)
    LAUNCH_SPLIT(128, 1);
  else if (num_experts == 128 && top_k == 2)
    LAUNCH_SPLIT(128, 2);
  else
    LAUNCH_SPLIT(128, 4);
#undef LAUNCH_SPLIT
}
