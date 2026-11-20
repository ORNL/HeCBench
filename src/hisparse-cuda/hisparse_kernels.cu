// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the SGLang project
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// HiSparse decode hot-buffer kernels
//
// The swap-in algorithm is a port of SGLang's hisparse
// load_cache_to_device_buffer kernel
// (python/sglang/jit_kernel/csrc/hisparse.cuh),
// adapted to vLLM addressing:
//  - tokens are keyed by their global KV slot id (block_table-converted
//    indexer output) instead of in-request positions, so no per-request
//    host-location table is needed: host pool row i is global slot i.
//  - each batch row owns `region_stride` device-global ids. Slots
//    [0, hot_size) are the LRU list. vLLM stores the row's newest token at
//    slot `hot_size` when region_stride == hot_size + 1; a region_stride
//    equal to hot_size has no extra slot, and every listed slot is eligible.
//
// The kernels only move bytes; they are dtype agnostic.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <cuda_runtime.h>
#include "hisparse_kernels.h"


namespace {

constexpr int kWarpSize = 32;
// Sentinel in the shared top-k scratch: entry already resolved (hit /
// newest / invalid), no miss handling needed.
constexpr int32_t kTokenDone = -1;
constexpr int32_t kHashEmpty = -1;

// First open-addressing slot for key. The multiplier is Knuth's
// 2654435761, the key is taken as an unsigned 32-bit pattern, and the
// result is that product modulo hash_size. hash_size must be positive.
__device__ __forceinline__ int32_t hash_slot(int32_t key, int32_t hash_size) {
  // Knuth multiplicative hash for the open-addressing table.
  return static_cast<int32_t>((static_cast<uint32_t>(key) * 2654435761u) %
                              static_cast<uint32_t>(hash_size));
}

// Copy row_bytes from src to dst. Lane lane_id of one warp copies every
// 32rd unit. If src, dst, and the length are all 16-byte aligned, units are
// uint4 loaded and stored with __ldcg/__stcg. Otherwise 4-byte alignment
// uses unsigned int, and any other alignment uses bytes. There is no tail:
// each path divides the length evenly.
__device__ __forceinline__ void copy_row_warp(int lane_id, const char* src,
                                              char* dst, int64_t row_bytes) {
  const auto alignment = reinterpret_cast<uintptr_t>(src) |
                         reinterpret_cast<uintptr_t>(dst) |
                         static_cast<uintptr_t>(row_bytes);
  if ((alignment & 15) == 0) {
    const int64_t num_vec = row_bytes / 16;
    const uint4* src4 = reinterpret_cast<const uint4*>(src);
    uint4* dst4 = reinterpret_cast<uint4*>(dst);
    for (int64_t j = lane_id; j < num_vec; j += kWarpSize) {
      __stcg(dst4 + j, __ldcg(src4 + j));
    }
    return;
  }

  if ((alignment & 3) == 0) {
    const int64_t num_words = row_bytes / 4;
    const unsigned int* src_words = reinterpret_cast<const unsigned int*>(src);
    unsigned int* dst_words = reinterpret_cast<unsigned int*>(dst);
    for (int64_t j = lane_id; j < num_words; j += kWarpSize) {
      __stcg(dst_words + j, __ldcg(src_words + j));
    }
    return;
  }

  for (int64_t j = lane_id; j < row_bytes; j += kWarpSize) {
    __stcg(dst + j, __ldcg(src + j));
  }
}

// Zero row_bytes at dst with one warp, using the same 16/4/1-byte alignment
// choice as copy_row_warp. The 16-byte path stores two zero uint64 values
// with st.global.cg; the narrower paths use __stcg.
__device__ __forceinline__ void zero_row_warp(int lane_id, char* dst,
                                              int64_t row_bytes) {
  const auto alignment =
      reinterpret_cast<uintptr_t>(dst) | static_cast<uintptr_t>(row_bytes);
  if ((alignment & 15) == 0) {
    const int64_t num_vec = row_bytes / 16;
    uint64_t* dst8 = reinterpret_cast<uint64_t*>(dst);
    for (int64_t j = lane_id; j < num_vec; j += kWarpSize) {
      uint64_t* d = dst8 + j * 2;
      asm volatile("st.global.cg.v2.b64 [%0],{%1,%2};" ::"l"(d), "l"(0ULL),
                   "l"(0ULL)
                   : "memory");
    }
    return;
  }

  if ((alignment & 3) == 0) {
    const int64_t num_words = row_bytes / 4;
    unsigned int* dst_words = reinterpret_cast<unsigned int*>(dst);
    for (int64_t j = lane_id; j < num_words; j += kWarpSize) {
      __stcg(dst_words + j, 0u);
    }
    return;
  }

  for (int64_t j = lane_id; j < row_bytes; j += kWarpSize) {
    __stcg(dst + j, static_cast<char>(0));
  }
}

// Byte address of row in a paged cache: page (row / block_size) starts at
// block_stride, and the row sits at (row % block_size) * row_bytes inside it.
// No bounds check.
__device__ __forceinline__ const char* cache_row_ptr(const char* cache,
                                                     int64_t row,
                                                     int32_t block_size,
                                                     int64_t block_stride,
                                                     int64_t row_bytes) {
  return cache + (row / block_size) * block_stride +
         (row % block_size) * row_bytes;
}

// Mutable overload of cache_row_ptr. Same address, no bounds check.
__device__ __forceinline__ char* cache_row_ptr(char* cache, int64_t row,
                                               int32_t block_size,
                                               int64_t block_stride,
                                               int64_t row_bytes) {
  return cache + (row / block_size) * block_stride +
         (row % block_size) * row_bytes;
}

// Copy one cache row with one warp. value_bytes <= 0 copies row_bytes as one
// contiguous span. Otherwise the value span and the following scale span
// (row_bytes - value_bytes) are copied separately; each is packed by offset
// within its page, so the two spans are not contiguous with each other.
__device__ __forceinline__ void copy_cache_row_warp(
    int lane_id, const char* src_cache, int64_t src_row, int32_t src_block_size,
    int64_t src_block_stride, char* dst_cache, int64_t dst_row,
    int32_t dst_block_size, int64_t dst_block_stride, int64_t row_bytes,
    int64_t value_bytes) {
  if (value_bytes <= 0) {
    copy_row_warp(lane_id,
                  cache_row_ptr(src_cache, src_row, src_block_size,
                                src_block_stride, row_bytes),
                  cache_row_ptr(dst_cache, dst_row, dst_block_size,
                                dst_block_stride, row_bytes),
                  row_bytes);
    return;
  }
  const int64_t scale_bytes = row_bytes - value_bytes;
  const int64_t src_offset = src_row % src_block_size;
  const int64_t dst_offset = dst_row % dst_block_size;
  const char* src_page =
      src_cache + (src_row / src_block_size) * src_block_stride;
  char* dst_page = dst_cache + (dst_row / dst_block_size) * dst_block_stride;
  copy_row_warp(lane_id, src_page + src_offset * value_bytes,
                dst_page + dst_offset * value_bytes, value_bytes);
  copy_row_warp(
      lane_id,
      src_page + src_block_size * value_bytes + src_offset * scale_bytes,
      dst_page + dst_block_size * value_bytes + dst_offset * scale_bytes,
      scale_bytes);
}

// Zero one cache row with one warp. value_bytes <= 0 zeros row_bytes as one
// span. Otherwise the value span and the scale span are zeroed separately,
// with the same page packing as copy_cache_row_warp.
__device__ __forceinline__ void zero_cache_row_warp(
    int lane_id, char* cache, int64_t row, int32_t block_size,
    int64_t block_stride, int64_t row_bytes, int64_t value_bytes) {
  if (value_bytes <= 0) {
    zero_row_warp(
        lane_id, cache_row_ptr(cache, row, block_size, block_stride, row_bytes),
        row_bytes);
    return;
  }
  const int64_t scale_bytes = row_bytes - value_bytes;
  const int64_t offset = row % block_size;
  char* page = cache + (row / block_size) * block_stride;
  zero_row_warp(lane_id, page + offset * value_bytes, value_bytes);
  zero_row_warp(lane_id, page + block_size * value_bytes + offset * scale_bytes,
                scale_bytes);
}

// In-place inclusive scan over s_data[offset, count) performed by warp 0,
// carrying `accumulator` across calls. Returns the running total.
__device__ __forceinline__ int warp_inclusive_scan(int32_t* s_data, int lane_id,
                                                   int offset, int count,
                                                   int accumulator) {
  int idx = lane_id + offset;
  int val = (idx < count) ? s_data[idx] : 0;
#pragma unroll
  for (int i = 1; i < 32; i *= 2) {
    int n = __shfl_up_sync(0xffffffff, val, i);
    if (lane_id >= i) val += n;
  }
  val += accumulator;
  if (idx < count) {
    s_data[idx] = val;
  }
  return __shfl_sync(0xffffffff, val, 31);
}

// Hot-cache row for slot. The block id is hot_block_table[row * table_stride
// + slot / block_size], and the row is that id times block_size plus
// slot % block_size. No bounds check.
__device__ __forceinline__ int64_t
get_physical_hot_row(const int32_t* hot_block_table, int32_t row,
                     int64_t table_stride, int32_t block_size, int32_t slot) {
  const int32_t block =
      hot_block_table[static_cast<int64_t>(row) * table_stride +
                      slot / block_size];
  return static_cast<int64_t>(block) * block_size + slot % block_size;
}

// Write physical_row to hot_indices[index]. When attention_indices is
// non-null, also write -1 if physical_row < 0, otherwise
// (physical_row / hot_block_size) * attention_block_stride +
// physical_row % hot_block_size. The product is computed in int64.
__device__ __forceinline__ void store_hot_index(
    int32_t* hot_indices, int32_t* attention_indices, int32_t index,
    int32_t physical_row, int32_t hot_block_size,
    int64_t attention_block_stride) {
  hot_indices[index] = physical_row;
  if (attention_indices != nullptr) {
    attention_indices[index] =
        physical_row < 0
            ? -1
            : (physical_row / hot_block_size) * attention_block_stride +
                  physical_row % hot_block_size;
  }
}

// One 1024-thread block per batch row. Translates that row's top-k token ids
// through the optional request, source, and resident tables, then assigns a
// hot slot to every id that still needs one. Resident rows and ids that are
// negative or past host_rows are finished immediately. The rest are matched
// against the LRU list: a hit keeps its slot, a miss takes the oldest
// evictable slot and records the new owner in device_global_indices. An LRU
// slot outside [0, region_stride) is not a hit and is not written. The list
// is rewritten as leftover evictable slots, then the new misses, then the
// hits. A negative state row writes -1 into its hot indexes and, when the
// pointers are non-null, its attention and resolved indexes, sets the valid
// and swap counts to 0 when those pointers are non-null, and returns.
// Requires hot_size >= top_k. Valid global ids must be unique within the row.
//
// Shared memory layout (int32 region followed by int16 region):
//   s_topk[top_k]            top-k global ids; reused as miss scratch
//   s_chunk_off[nbc + 1]     prefix sums for hit (then miss) compaction
//   s_evict_off[nbc + 1]     prefix sums for evictable compaction
//   s_hash_keys[hash_size]   open addressing: global id -> top-k index
//   s_counters[3]            hits, phase-1 resolved, valid count
//   s_lru_out[hot_size]      int16, compacted slots: [hits fwd | evict bwd]
//   s_hash_vals[hash_size]   int16 hash values (top-k index)
__global__ __launch_bounds__(1024) void hisparse_resolve_residency_kernel(
    const int32_t* __restrict__ hot_block_table,  // [max_rows, hot_blocks]
    const int32_t* __restrict__ global_indices,   // global or request-relative
    const int32_t* __restrict__ request_ids,      // [num_rows] or nullptr
    const int32_t* __restrict__ source_block_table,  // [num_reqs, max_blocks]
    const int32_t* __restrict__ resident_block_table,
    int32_t* __restrict__ resolved_global_indices,    // [num_rows, top_k]
    int32_t* __restrict__ valid_counts,               // [num_rows] or nullptr
    int32_t* __restrict__ swap_host_physical_rows,    // [num_rows, top_k]
    int32_t* __restrict__ swap_device_physical_rows,  // [num_rows, top_k]
    int32_t* __restrict__ swap_counts,                // [num_rows]
    int32_t* __restrict__ hot_indices,                // [num_rows, top_k]
    int32_t* __restrict__ attention_indices,          // [num_rows, top_k]
    int32_t* __restrict__ miss_mask,  // [num_rows, top_k] or nullptr
    int32_t* __restrict__ device_global_indices,  // [max_rows, region_stride]
    int16_t* __restrict__ lru_slots,              // [max_rows, hot_size]
    unsigned long long* __restrict__ stats,       // [2] hits,misses or nullptr
    const int32_t* __restrict__ request_state_indices,  // [num_requests] or
                                                        // nullptr
    const int32_t request_state_count, const int64_t host_rows,
    const int64_t hot_table_stride, const int32_t hot_block_size,
    const int32_t top_k, const int32_t hot_size, const int32_t hash_size,
    const int64_t region_stride, const int64_t attention_block_stride,
    const int64_t source_bt_stride, const int32_t source_num_reqs,
    const int32_t source_num_blocks, const int32_t source_block_size,
    const int64_t resident_bt_stride, const int32_t resident_num_reqs,
    const int32_t resident_num_blocks, const int32_t resident_block_size,
    const int32_t resident_null_block, const int64_t input_row_stride,
    const int64_t attention_row_stride, const int64_t valid_count_stride) {
  const int NUM_WARPS = blockDim.x / kWarpSize;
  const int num_buffer_chunks = (hot_size + kWarpSize - 1) / kWarpSize;
  const int num_token_chunks = (top_k + kWarpSize - 1) / kWarpSize;

  const int batch_row = blockIdx.x;
  const int request_row =
      request_ids != nullptr ? request_ids[batch_row] : batch_row;
  const int state_row =
      request_state_indices != nullptr && request_row >= 0 &&
              request_row < request_state_count
          ? request_state_indices[request_row]
          : (request_state_indices == nullptr ? request_row : -1);
  // V2 publishes -1 for graph-capture padding rows.
  if (state_row < 0) {
    for (int i = threadIdx.x; i < top_k; i += blockDim.x) {
      const int64_t index = static_cast<int64_t>(batch_row) * top_k + i;
      hot_indices[index] = -1;
      if (attention_indices != nullptr) {
        attention_indices[static_cast<int64_t>(batch_row) *
                              attention_row_stride + i] = -1;
      }
      if (resolved_global_indices != nullptr) {
        resolved_global_indices[index] = -1;
      }
    }
    if (valid_counts != nullptr && threadIdx.x == 0) {
      valid_counts[static_cast<int64_t>(batch_row) * valid_count_stride] = 0;
    }
    if (swap_counts != nullptr && threadIdx.x == 0) {
      swap_counts[batch_row] = 0;
    }
    return;
  }
  const int tid = threadIdx.x;
  const int warp_id = tid / kWarpSize;
  const int lane_id = tid % kWarpSize;
  const unsigned int lanes_before = ((unsigned int)1 << lane_id) - 1;

  const int32_t* row_topk =
      global_indices + static_cast<int64_t>(batch_row) * input_row_stride;
  int32_t* row_out = hot_indices + static_cast<int64_t>(batch_row) * top_k;
  int32_t* row_attention =
      attention_indices != nullptr
          ? attention_indices +
                static_cast<int64_t>(batch_row) * attention_row_stride
          : nullptr;
  int32_t* row_miss = (miss_mask != nullptr)
                          ? miss_mask + static_cast<int64_t>(batch_row) * top_k
                          : nullptr;
  int32_t* row_dgi =
      device_global_indices + static_cast<int64_t>(state_row) * region_stride;
  int16_t* row_lru = lru_slots + static_cast<int64_t>(state_row) * hot_size;

  extern __shared__ char smem_raw[];
  int32_t* s_topk = reinterpret_cast<int32_t*>(smem_raw);
  int32_t* s_chunk_off = s_topk + top_k;
  int32_t* s_evict_off = s_chunk_off + (num_buffer_chunks + 1);
  int32_t* s_hash_keys = s_evict_off + (num_buffer_chunks + 1);
  int32_t* s_counters = s_hash_keys + hash_size;
  int16_t* s_lru_out = reinterpret_cast<int16_t*>(s_counters + 3);
  int16_t* s_hash_vals = s_lru_out + hot_size;

  if (tid < 3) {
    s_counters[tid] = 0;
  }
  __syncthreads();

  // Phase 1: translate request-relative positions and resolve resident rows.
  for (int i = tid; i < top_k; i += blockDim.x) {
    const int32_t token_index = row_topk[i];
    int32_t g = token_index;
    int32_t resident_row = -1;
    if (source_block_table != nullptr) {
      const int32_t request_id = request_ids[batch_row];
      const int32_t source_block =
          token_index >= 0 ? token_index / source_block_size : -1;
      if (request_id >= 0 && request_id < source_num_reqs &&
          source_block >= 0 && source_block < source_num_blocks) {
        const int32_t physical_block =
            source_block_table[static_cast<int64_t>(request_id) *
                                   source_bt_stride +
                               source_block];
        g = physical_block > 0 ? physical_block * source_block_size +
                                     token_index % source_block_size
                               : -1;
      } else {
        g = -1;
      }
      if (resident_block_table != nullptr) {
        const int32_t resident_block =
            token_index >= 0 ? token_index / resident_block_size : -1;
        if (request_id >= 0 && request_id < resident_num_reqs &&
            resident_block >= 0 && resident_block < resident_num_blocks) {
          const int32_t physical_block =
              resident_block_table[static_cast<int64_t>(request_id) *
                                       resident_bt_stride +
                                   resident_block];
          if (physical_block != resident_null_block && physical_block >= 0) {
            resident_row = physical_block * resident_block_size +
                           token_index % resident_block_size;
          }
        }
      }
    }
    if (g >= host_rows) {
      g = -1;
    }
    if (resolved_global_indices != nullptr) {
      resolved_global_indices[static_cast<int64_t>(batch_row) * top_k + i] = g;
    }
    if (resident_row >= 0 || g >= 0) atomicAdd(&s_counters[2], 1);
    if (row_miss != nullptr) row_miss[i] = 0;
    if (resident_row >= 0) {
      store_hot_index(row_out, row_attention, i, resident_row, hot_block_size,
                      attention_block_stride);
      s_topk[i] = kTokenDone;
      atomicAdd(&s_counters[1], 1);
    } else if (g < 0) {
      store_hot_index(row_out, row_attention, i, -1, hot_block_size,
                      attention_block_stride);
      s_topk[i] = kTokenDone;
      atomicAdd(&s_counters[1], 1);
    } else {
      s_topk[i] = g;
    }
  }
  __syncthreads();
  if (valid_counts != nullptr && tid == 0) {
    valid_counts[static_cast<int64_t>(batch_row) * valid_count_stride] =
        s_counters[2];
  }
  // Fully resident rows need only request-relative page translation. Avoid
  // scanning or rewriting the hot LRU when no selected row can consult it.
  if (s_counters[1] == top_k) {
    if (tid == 0 && swap_counts != nullptr) {
      swap_counts[batch_row] = 0;
    }
    return;
  }

  for (int i = tid; i < hash_size; i += blockDim.x) {
    s_hash_keys[i] = kHashEmpty;
  }
  for (int i = tid; i < num_buffer_chunks + 1; i += blockDim.x) {
    s_chunk_off[i] = 0;
    s_evict_off[i] = 0;
  }
  __syncthreads();

  for (int i = tid; i < top_k; i += blockDim.x) {
    const int32_t g = s_topk[i];
    if (g == kTokenDone) {
      continue;
    }
    int slot = hash_slot(g, hash_size);
    while (true) {
      const int32_t old = atomicCAS(&s_hash_keys[slot], kHashEmpty, g);
      if (old == kHashEmpty || old == g) {
        s_hash_vals[slot] = static_cast<int16_t>(i);
        break;
      }
      slot = (slot + 1) % hash_size;
    }
  }
  __syncthreads();

  // Phase 2: walk hot slots in LRU order, classify hit / evictable, and
  // compact them (hits forward, evictables backward) into s_lru_out.
  const int iters_buffer = (num_buffer_chunks + NUM_WARPS - 1) / NUM_WARPS;
  int total_hit_count = 0;
  int total_evict_count = 0;
  for (int iter = 0; iter < iters_buffer; iter++) {
    const int chunk_idx = warp_id + iter * NUM_WARPS;
    const bool has_valid_chunk = chunk_idx < num_buffer_chunks;

    const int pos = chunk_idx * kWarpSize + lane_id;
    const bool has_valid_pos = has_valid_chunk && (pos < hot_size);
    const int16_t slot = has_valid_pos ? row_lru[pos] : int16_t(-1);
    // Corruption tripwire: lru/dgi are long-lived device state; if some
    // external writer (e.g. stray RDMA into reused VRAM) corrupts a slot
    // out of [0, region_stride), treat it as no-hit so it degrades to a re-miss
    // instead of an unbounded dgi read. Phase 3 skips the dgi write for that
    // slot, and the gather kernels skip a negative device slot.
    const int32_t cached_g =
        (slot >= 0 && slot < region_stride) ? row_dgi[slot] : -1;

    int found_topk_idx = -1;
    if (cached_g >= 0) {
      int h = hash_slot(cached_g, hash_size);
      while (true) {
        const int32_t k = s_hash_keys[h];
        if (k == cached_g) {
          found_topk_idx = static_cast<int32_t>(s_hash_vals[h]);
          break;
        }
        if (k == kHashEmpty) break;
        h = (h + 1) % hash_size;
      }
    }
    const bool is_hit = found_topk_idx >= 0;
    const bool is_evictable = has_valid_pos && !is_hit;

    if (is_hit) {
      s_topk[found_topk_idx] = kTokenDone;
      store_hot_index(row_out, row_attention, found_topk_idx,
                      static_cast<int32_t>(get_physical_hot_row(
                          hot_block_table, request_row, hot_table_stride,
                          hot_block_size, slot)),
                      hot_block_size, attention_block_stride);
    }

    int local_hit_off = 0;
    int local_evict_off = 0;
    if (has_valid_chunk) {
      const unsigned int hit_mask = __ballot_sync(0xFFFFFFFF, is_hit);
      const unsigned int evict_mask = __ballot_sync(0xFFFFFFFF, is_evictable);
      local_hit_off = __popc(hit_mask & lanes_before);
      local_evict_off = __popc(evict_mask & lanes_before);
      if (lane_id == 0) {
        s_chunk_off[chunk_idx + 1] = __popc(hit_mask);
        s_evict_off[chunk_idx + 1] = __popc(evict_mask);
      }
    }
    __syncthreads();

    if (warp_id == 0) {
      total_hit_count =
          warp_inclusive_scan(s_chunk_off, lane_id, chunk_idx + 1,
                              num_buffer_chunks + 1, total_hit_count);
      total_evict_count =
          warp_inclusive_scan(s_evict_off, lane_id, chunk_idx + 1,
                              num_buffer_chunks + 1, total_evict_count);
      if (tid == 0) {
        s_counters[0] = total_hit_count;
      }
    }
    __syncthreads();

    if (is_hit) {
      const int off = s_chunk_off[chunk_idx] + local_hit_off;
      s_lru_out[off] = slot;
    }
    if (is_evictable) {
      const int off = s_evict_off[chunk_idx] + local_evict_off;
      s_lru_out[hot_size - 1 - off] = slot;
    }
  }
  __syncthreads();

  // Reset prefix sums for the miss compaction (token chunks <= buffer
  // chunks because hot_size >= top_k).
  for (int i = tid; i < num_token_chunks + 1; i += blockDim.x) {
    s_chunk_off[i] = 0;
  }
  __syncthreads();

  // Phase 3: compact misses, assign them eviction slots (oldest first) and
  // record the new ownership in device_global_indices.
  const int iters_token = (num_token_chunks + NUM_WARPS - 1) / NUM_WARPS;
  int miss_running_total = 0;
  for (int iter = 0; iter < iters_token; iter++) {
    const int chunk_idx = warp_id + iter * NUM_WARPS;
    const bool has_valid_chunk = chunk_idx < num_token_chunks;

    const int i = chunk_idx * kWarpSize + lane_id;
    const bool has_valid_token = has_valid_chunk && (i < top_k);

    int32_t g = 0;
    bool is_miss = false;
    if (has_valid_token) {
      is_miss = s_topk[i] != kTokenDone;
      if (is_miss) {
        g = s_topk[i];
      }
    }

    int local_miss_off = 0;
    if (has_valid_chunk) {
      const unsigned int miss_mask = __ballot_sync(0xFFFFFFFF, is_miss);
      local_miss_off = __popc(miss_mask & lanes_before);
      if (lane_id == 0) {
        s_chunk_off[chunk_idx + 1] = __popc(miss_mask);
      }
    }
    __syncthreads();

    if (warp_id == 0) {
      miss_running_total =
          warp_inclusive_scan(s_chunk_off, lane_id, chunk_idx + 1,
                              num_token_chunks + 1, miss_running_total);
    }
    __syncthreads();

    if (is_miss) {
      const int m = s_chunk_off[chunk_idx] + local_miss_off;
      const int16_t evict_slot = s_lru_out[hot_size - 1 - m];
      if (evict_slot < 0 || evict_slot >= region_stride) {
        // Corruption tripwire: an out-of-range slot (corrupted lru state,
        // see phase 2) must not become a hot-cache or dgi write. Resolve the
        // entry as invalid (-1, masked by attention, not in the miss set).
        // The compact record keeps the host id and a device slot of -1, and
        // the gather kernels skip that copy.
        store_hot_index(row_out, row_attention, i, -1, hot_block_size,
                        attention_block_stride);
        if (swap_host_physical_rows != nullptr) {
          const int64_t compact_index =
              static_cast<int64_t>(batch_row) * top_k + m;
          swap_host_physical_rows[compact_index] = g;
          swap_device_physical_rows[compact_index] = -1;
        }
      } else {
        // Reuse s_topk as compacted miss scratch: m < i always holds (done
        // entries are skipped), so writes never overrun pending reads.
        s_topk[m] = g;
        const int32_t physical_row = static_cast<int32_t>(
            get_physical_hot_row(hot_block_table, request_row, hot_table_stride,
                                 hot_block_size, evict_slot));
        store_hot_index(row_out, row_attention, i, physical_row, hot_block_size,
                        attention_block_stride);
        if (swap_host_physical_rows != nullptr) {
          const int64_t compact_index =
              static_cast<int64_t>(batch_row) * top_k + m;
          swap_host_physical_rows[compact_index] = g;
          swap_device_physical_rows[compact_index] = physical_row;
        }
        if (row_miss != nullptr) row_miss[i] = 1;
        row_dgi[evict_slot] = g;
      }
    }
  }
  __syncthreads();

  const int total_hits = s_counters[0];
  const int total_misses = top_k - total_hits - s_counters[1];
  if (swap_counts != nullptr && tid == 0) {
    swap_counts[batch_row] = total_misses;
  }
  if (stats != nullptr && tid == 0) {
    atomicAdd(&stats[0], static_cast<unsigned long long>(total_hits));
    atomicAdd(&stats[1], static_cast<unsigned long long>(total_misses));
  }

  // Phase 4: write back the LRU order: stale evictables at the front,
  // freshly loaded misses next, then hits at MRU.
  const int total_evictable = hot_size - total_hits;
  const int remaining_evictable = total_evictable - total_misses;
  for (int i = tid; i < hot_size; i += blockDim.x) {
    if (i < remaining_evictable) {
      row_lru[i] = s_lru_out[hot_size - 1 - total_misses - i];
    } else if (i < remaining_evictable + total_misses) {
      row_lru[i] = s_lru_out[hot_size - 1 - (i - remaining_evictable)];
    } else {
      row_lru[i] = s_lru_out[i - remaining_evictable - total_misses];
    }
  }
}

// Replay an existing plan into this layer's hot cache. blockIdx.x is the
// batch row and columns are striped across blockIdx.y and the warps. A row
// whose request_state_indices entry is negative is padding: attention is set
// to -1 and nothing is copied. Every other column gets an attention id from
// its hot index. A column is copied only when its miss flag is nonzero, both
// indexes are in range, and the host index is below host_rows; a host index
// at or past host_rows zeros the hot slot instead. miss_mask is required.
__global__ void hisparse_gather_plan_kernel(
    const char* __restrict__ host_cache,         // [host_rows, row_bytes]
    char* __restrict__ hot_cache,                // [hot_rows, row_bytes]
    const int32_t* __restrict__ global_indices,  // [num_rows, top_k]
    const int32_t* __restrict__ hot_indices,  // [num_rows, top_k] abs hot rows
    const int32_t* __restrict__ miss_mask,    // [num_rows, top_k]
    int32_t* __restrict__ attention_indices,  // [num_rows, top_k]
    const int32_t* __restrict__ request_state_indices,  // [num_rows] or nullptr
    const int64_t host_rows, const int32_t host_block_size,
    const int64_t host_block_stride, const int64_t hot_rows,
    const int64_t row_bytes, const int64_t row_value_bytes,
    const int64_t hot_block_stride, const int32_t hot_block_size,
    const int32_t top_k, const int64_t attention_block_stride) {
  const int NUM_WARPS = blockDim.x / kWarpSize;
  // Columns are interleaved across gridDim.y blocks per row so few-row
  // launches (local-prefill staging's single-row layout, small decode
  // batches) still fill the device with outstanding host reads instead of
  // starving on one block per row.
  const int row = blockIdx.x;
  const bool is_padding =
      request_state_indices != nullptr && request_state_indices[row] < 0;
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  const int col_start = blockIdx.y * NUM_WARPS + warp_id;
  const int col_stride = gridDim.y * NUM_WARPS;
  const int64_t base = static_cast<int64_t>(row) * top_k;
  for (int col = col_start; col < top_k; col += col_stride) {
    const int32_t dst = hot_indices[base + col];
    if (lane_id == 0 && attention_indices != nullptr) {
      attention_indices[base + col] =
          is_padding
              ? -1
              : (dst < 0 ? -1
                         : (dst / hot_block_size) * attention_block_stride +
                               dst % hot_block_size);
    }
    if (is_padding) {
      continue;
    }
    if (miss_mask[base + col] == 0) {
      continue;
    }
    const int32_t g = global_indices[base + col];
    if (g < 0 || dst < 0 || dst >= hot_rows) {
      continue;
    }
    if (g < host_rows) {
      copy_cache_row_warp(lane_id, host_cache, g, host_block_size,
                          host_block_stride, hot_cache, dst, hot_block_size,
                          hot_block_stride, row_bytes, row_value_bytes);
    } else {
      // No source row for g: zero the planned slot rather than serving
      // whatever bytes it held.
      zero_cache_row_warp(lane_id, hot_cache, dst, hot_block_size,
                          hot_block_stride, row_bytes, row_value_bytes);
    }
  }
}

// Copy the compact miss list produced by resolve. For each batch row the
// count is clamped to [0, top_k], and those leading compact entries are
// striped across blockIdx.y and the warps. An entry is skipped when the host
// index is negative or the hot index is out of range. A host index below
// host_rows is copied; one at or past host_rows zeros the hot slot.
__global__ void hisparse_gather_compact_kernel(
    const char* __restrict__ host_cache, char* __restrict__ hot_cache,
    const int32_t* __restrict__ miss_global_indices,
    const int32_t* __restrict__ miss_hot_indices,
    const int32_t* __restrict__ miss_counts, const int64_t host_rows,
    const int32_t host_block_size, const int64_t host_block_stride,
    const int64_t hot_rows, const int64_t row_bytes,
    const int64_t row_value_bytes, const int64_t hot_block_stride,
    const int32_t hot_block_size, const int32_t top_k) {
  const int NUM_WARPS = blockDim.x / kWarpSize;
  const int row = blockIdx.x;
  const int warp_id = threadIdx.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  const int miss_count = min(max(miss_counts[row], 0), top_k);
  const int col_start = blockIdx.y * NUM_WARPS + warp_id;
  const int col_stride = gridDim.y * NUM_WARPS;
  const int64_t base = static_cast<int64_t>(row) * top_k;
  for (int col = col_start; col < miss_count; col += col_stride) {
    const int32_t g = miss_global_indices[base + col];
    const int32_t dst = miss_hot_indices[base + col];
    if (g < 0 || dst < 0 || dst >= hot_rows) {
      continue;
    }
    if (g < host_rows) {
      copy_cache_row_warp(lane_id, host_cache, g, host_block_size,
                          host_block_stride, hot_cache, dst, hot_block_size,
                          hot_block_stride, row_bytes, row_value_bytes);
    } else {
      zero_cache_row_warp(lane_id, hot_cache, dst, hot_block_size,
                          hot_block_stride, row_bytes, row_value_bytes);
    }
  }
}

// One warp per item, grid-stride over num_items. Copy src_cache[src_indices[i]]
// into host_cache[dst_slots[i]]. Skip the item when the source is outside
// [0, src_rows) or the destination is outside [0, host_rows).
__global__ void hisparse_backup_kernel(
    const char* __restrict__ src_cache, const int64_t* __restrict__ src_indices,
    char* __restrict__ host_cache, const int64_t* __restrict__ dst_slots,
    const int64_t row_bytes, const int64_t row_value_bytes,
    const int64_t src_block_stride, const int32_t src_block_size,
    const int64_t host_block_stride, const int32_t host_block_size,
    const int32_t num_items, const int64_t src_rows, const int64_t host_rows) {
  const int NUM_WARPS = blockDim.x / kWarpSize;
  const int lane_id = threadIdx.x % kWarpSize;
  const int warp_id = blockIdx.x * NUM_WARPS + threadIdx.x / kWarpSize;
  const int total_warps = gridDim.x * NUM_WARPS;

  for (int i = warp_id; i < num_items; i += total_warps) {
    const int64_t s = src_indices[i];
    const int64_t d = dst_slots[i];
    if (s < 0 || s >= src_rows || d < 0 || d >= host_rows) {
      continue;
    }
    copy_cache_row_warp(lane_id, src_cache, s, src_block_size, src_block_stride,
                        host_cache, d, host_block_size, host_block_stride,
                        row_bytes, row_value_bytes);
  }
}

// One block per token. Look up that token's request and state row, then set
// every device_global_indices entry in the state row that equals the written
// slot to -1. Return without writing when the token is past num_tokens, the
// request is outside [0, num_request_ids), the state is outside
// [0, num_state_rows), or the written slot is negative.
__global__ void hisparse_invalidate_written_slots_kernel(
    int32_t* __restrict__ device_global_indices,
    const int32_t* __restrict__ request_state_indices,
    const int32_t* __restrict__ req_id_per_token,
    const int64_t* __restrict__ written_slots, const int64_t num_tokens,
    const int64_t num_request_ids, const int64_t num_state_rows,
    const int64_t region_stride) {
  const int64_t token_idx = blockIdx.x;
  if (token_idx >= num_tokens) {
    return;
  }
  const int32_t req_idx = req_id_per_token[token_idx];
  if (req_idx < 0 || req_idx >= num_request_ids) {
    return;
  }
  const int32_t state_idx = request_state_indices[req_idx];
  const int64_t written_slot = written_slots[token_idx];
  if (state_idx < 0 || state_idx >= num_state_rows || written_slot < 0) {
    return;
  }
  int32_t* row = device_global_indices + state_idx * region_stride;
  for (int64_t offset = threadIdx.x; offset < region_stride;
       offset += blockDim.x) {
    if (row[offset] == written_slot) {
      row[offset] = -1;
    }
  }
}


}  // namespace

namespace {

// Return cudaGetLastError(). On failure, print what and the CUDA string.
// A success leaves the sticky error cleared.
cudaError_t launch_error(const char* what) {
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(err));
  }
  return err;
}

}  // namespace

// Dynamic shared memory for one resolve block when hot_size >= top_k.
// int32 region: top_k ids, two prefix arrays of (ceil(hot_size/32)+1),
// a hash table of 2*top_k keys, and 3 counters. int16 region: hot_size
// compacted slots and 2*top_k hash values. Does not check the arguments.
size_t hisparse_resolve_smem_bytes(int32_t top_k, int32_t hot_size) {
  const int hash_size = 2 * top_k;
  const int num_buffer_chunks = (hot_size + 32 - 1) / 32;
  return sizeof(int32_t) * (static_cast<size_t>(top_k) +
                            2 * (static_cast<size_t>(num_buffer_chunks) + 1) +
                            static_cast<size_t>(hash_size) + 3) +
         sizeof(int16_t) *
             (static_cast<size_t>(hot_size) + static_cast<size_t>(hash_size));
}

// Launch one resolve block per batch row on stream. Return success without
// launching when num_rows or top_k is 0. Return cudaErrorInvalidValue, after
// printing the reason, when a source block table is set without request_ids,
// hot_size < top_k, hot_size > 32768, or the dynamic shared memory does not
// fit in an int. Above 48 KiB, raise this device's opt-in once to the needed
// size and remember it for later calls on the same device; that memory is
// not safe across threads or cudaDeviceReset. Then launch and return the
// launch error.
cudaError_t hisparse_resolve_residency_launch(const HisparseResolveArgs& args,
                                              cudaStream_t stream) {
  if (args.num_rows == 0 || args.top_k == 0) return cudaSuccess;
  if (args.source_block_table != nullptr && args.request_ids == nullptr) {
    fprintf(stderr, "HiSparse source_block_table requires request_ids\n");
    return cudaErrorInvalidValue;
  }
  if (args.hot_size < args.top_k) {
    fprintf(stderr, "HiSparse requires hot_size >= top_k (%d < %d)\n",
            args.hot_size, args.top_k);
    return cudaErrorInvalidValue;
  }
  // LRU slots and hash values are stored as int16.
  if (args.hot_size > std::numeric_limits<int16_t>::max() + 1) {
    fprintf(stderr, "HiSparse hot_size %d exceeds the int16 slot range\n",
            args.hot_size);
    return cudaErrorInvalidValue;
  }
  constexpr int kBlockSize = 1024;
  const int hash_size = 2 * args.top_k;
  const size_t smem_bytes =
      hisparse_resolve_smem_bytes(args.top_k, args.hot_size);
  auto kernel = hisparse_resolve_residency_kernel;
  // Opt-in size already applied on the current device, so repeat launches skip
  // the host call. Assumes a single host thread and no cudaDeviceReset.
  static int configured_device = -1;
  static size_t configured_smem = 48 * 1024;
  if (smem_bytes > 48 * 1024) {
    if (smem_bytes > static_cast<size_t>(std::numeric_limits<int>::max())) {
      fprintf(stderr, "HiSparse shared memory %zu does not fit in int\n",
              smem_bytes);
      return cudaErrorInvalidValue;
    }
    int device = 0;
    const cudaError_t device_error = cudaGetDevice(&device);
    if (device_error != cudaSuccess) return device_error;
    if (device != configured_device) {
      configured_device = device;
      configured_smem = 48 * 1024;
    }
    if (smem_bytes > configured_smem) {
      const cudaError_t attribute_error = cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(smem_bytes));
      if (attribute_error != cudaSuccess) {
        fprintf(stderr, "failed to configure HiSparse shared memory: %s\n",
                cudaGetErrorString(attribute_error));
        return attribute_error;
      }
      configured_smem = smem_bytes;
    }
  }
  kernel<<<args.num_rows, kBlockSize, smem_bytes, stream>>>(
      args.hot_block_table, args.global_indices, args.request_ids,
      args.source_block_table, args.resident_block_table,
      args.resolved_global_indices, args.valid_counts,
      args.swap_host_physical_rows, args.swap_device_physical_rows,
      args.swap_counts, args.hot_indices, args.attention_indices,
      args.miss_mask, args.device_global_indices, args.lru_slots, args.stats,
      args.request_state_indices, args.request_state_count, args.host_rows,
      args.hot_table_stride, args.hot_block_size, args.top_k, args.hot_size,
      hash_size, args.region_stride, args.attention_block_stride,
      args.source_bt_stride, args.source_num_reqs, args.source_num_blocks,
      args.source_block_size, args.resident_bt_stride, args.resident_num_reqs,
      args.resident_num_blocks, args.resident_block_size,
      args.resident_null_block, args.input_row_stride, args.attention_row_stride,
      args.valid_count_stride);
  return launch_error("HiSparse residency kernel");
}

// Launch gather_plan with 1024-thread blocks on stream. The y dimension is
// min(ceil(top_k / 32), max(1, 256 / num_rows)); columns beyond that are
// still visited by the kernel's stride. Return success without launching
// when num_rows or top_k is 0, then return the launch error.
cudaError_t hisparse_gather_plan_launch(
    const char* host_cache, char* hot_cache, const int32_t* global_indices,
    const int32_t* hot_indices, const int32_t* miss_mask,
    int32_t* attention_indices, const int32_t* request_state_indices,
    int64_t host_rows, int32_t host_block_size, int64_t host_block_stride,
    int64_t hot_rows, int64_t row_bytes, int64_t row_value_bytes,
    int64_t hot_block_stride, int32_t hot_block_size, int32_t num_rows,
    int32_t top_k, int64_t attention_block_stride, cudaStream_t stream) {
  if (num_rows == 0 || top_k == 0) return cudaSuccess;
  constexpr int kBlockSize = 1024;
  constexpr int kNumWarps = kBlockSize / 32;
  constexpr int kTargetBlocks = 256;
  const int max_chunks = std::max(1, (top_k + kNumWarps - 1) / kNumWarps);
  const int num_chunks =
      std::min(max_chunks, std::max(1, kTargetBlocks / num_rows));
  const dim3 grid(num_rows, num_chunks);
  hisparse_gather_plan_kernel<<<grid, kBlockSize, 0, stream>>>(
      host_cache, hot_cache, global_indices, hot_indices, miss_mask,
      attention_indices, request_state_indices, host_rows, host_block_size,
      host_block_stride, hot_rows, row_bytes, row_value_bytes, hot_block_stride,
      hot_block_size, top_k, attention_block_stride);
  return launch_error("HiSparse gather-plan kernel");
}

// Launch gather_compact with 512-thread blocks on stream. The y dimension is
// min(8, max(1, 64 / num_rows)). Return success without launching when
// num_rows or top_k is 0, then return the launch error.
cudaError_t hisparse_gather_compact_launch(
    const char* host_cache, char* hot_cache, const int32_t* miss_global_indices,
    const int32_t* miss_hot_indices, const int32_t* miss_counts,
    int64_t host_rows, int32_t host_block_size, int64_t host_block_stride,
    int64_t hot_rows, int64_t row_bytes, int64_t row_value_bytes,
    int64_t hot_block_stride, int32_t hot_block_size, int32_t num_rows,
    int32_t top_k, cudaStream_t stream) {
  if (num_rows == 0 || top_k == 0) return cudaSuccess;
  constexpr int kBlockSize = 512;
  constexpr int kTargetBlocks = 64;
  const int num_chunks = std::min(8, std::max(1, kTargetBlocks / num_rows));
  const dim3 grid(num_rows, num_chunks);
  hisparse_gather_compact_kernel<<<grid, kBlockSize, 0, stream>>>(
      host_cache, hot_cache, miss_global_indices, miss_hot_indices, miss_counts,
      host_rows, host_block_size, host_block_stride, hot_rows, row_bytes,
      row_value_bytes, hot_block_stride, hot_block_size, top_k);
  return launch_error("HiSparse gather-compact kernel");
}

// Launch backup with 256-thread blocks, one warp per item, rounding the grid
// up so every item has a warp. Return success without launching when
// num_items is 0.
cudaError_t hisparse_backup_launch(const char* src_cache, const int64_t* src_indices,
                            char* host_cache, const int64_t* dst_slots,
                            int64_t row_bytes, int64_t row_value_bytes,
                            int64_t src_block_stride, int32_t src_block_size,
                            int64_t host_block_stride, int32_t host_block_size,
                            int32_t num_items, int64_t src_rows,
                            int64_t host_rows, cudaStream_t stream) {
  if (num_items == 0) return cudaSuccess;
  constexpr int kBlockSize = 256;
  constexpr int kNumWarps = kBlockSize / 32;
  const int grid = (num_items + kNumWarps - 1) / kNumWarps;
  hisparse_backup_kernel<<<grid, kBlockSize, 0, stream>>>(
      src_cache, src_indices, host_cache, dst_slots, row_bytes, row_value_bytes,
      src_block_stride, src_block_size, host_block_stride, host_block_size,
      num_items, src_rows, host_rows);
  return launch_error("HiSparse backup kernel");
}

// Launch one 256-thread block per token. Return success without launching
// when num_tokens, num_state_rows, or region_stride is 0.
cudaError_t hisparse_invalidate_written_slots_launch(
    int32_t* device_global_indices, const int32_t* request_state_indices,
    const int32_t* req_id_per_token, const int64_t* written_slots,
    int64_t num_tokens, int64_t num_request_ids, int64_t num_state_rows,
    int64_t region_stride, cudaStream_t stream) {
  if (num_tokens == 0 || num_state_rows == 0 || region_stride == 0) {
    return cudaSuccess;
  }
  constexpr int kBlockSize = 256;
  hisparse_invalidate_written_slots_kernel<<<static_cast<int>(num_tokens),
                                             kBlockSize, 0, stream>>>(
      device_global_indices, request_state_indices, req_id_per_token,
      written_slots, num_tokens, num_request_ids, num_state_rows, region_stride);
  return launch_error("HiSparse invalidation kernel");
}
