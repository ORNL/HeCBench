// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the SGLang project
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// HiSparse decode hot-buffer kernels for sparse MLA, SYCL port.
//
// A CUDA warp maps to a SYCL sub-group. The launchers request the largest
// size the device lists (4, 8, 16, 32, or 64), which is what the other SYCL
// benchmarks select; without the request the compiler uses the smallest,
// 16 on Intel GPUs that also offer 32. Lane strides, chunk sizes,
// ballot-style compaction, and the sub-group-0 prefix scan read that width
// at run time, and the host sizes local memory and grids from the same
// value.
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

#include "hisparse_kernels.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>

namespace {

// Sentinel in the local top-k scratch: entry already resolved (hit /
// newest / invalid), no miss handling needed.
constexpr int32_t kTokenDone = -1;
constexpr int32_t kHashEmpty = -1;

using local_atomic_int =
    sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                     sycl::memory_scope::work_group,
                     sycl::access::address_space::local_space>;
using global_atomic_u64 =
    sycl::atomic_ref<unsigned long long, sycl::memory_order::relaxed,
                     sycl::memory_scope::device,
                     sycl::access::address_space::global_space>;

// 16-byte copy unit. sycl::uint4 assignment is lowered to two strided 8-byte
// stores on nvptx, which turns writes into the host pool into partial PCIe
// sectors and runs the backup ~20x slower; a clang vector stays one 128-bit
// access.
#if defined(__clang__)
typedef unsigned int copy_unit16 __attribute__((ext_vector_type(4)));
#else
using copy_unit16 = sycl::uint4;
#endif

// First open-addressing slot for key. The multiplier is Knuth's
// 2654435761, the key is taken as an unsigned 32-bit pattern, and the
// result is that product modulo hash_size. hash_size must be positive.
inline int32_t hash_slot(int32_t key, int32_t hash_size) {
  return static_cast<int32_t>((static_cast<uint32_t>(key) * 2654435761u) %
                              static_cast<uint32_t>(hash_size));
}

// Copy row_bytes from src to dst. Lane lane_id of a sub-group of width
// sg_size copies every sg_size-th unit. If src, dst, and the length are all
// 16-byte aligned, units are copy_unit16. Otherwise 4-byte alignment uses
// unsigned int, and any other alignment uses bytes. There is no tail: each
// path divides the length evenly.
inline void copy_row_warp(int lane_id, int sg_size, const char* src, char* dst,
                          int64_t row_bytes) {
  const auto alignment = reinterpret_cast<uintptr_t>(src) |
                         reinterpret_cast<uintptr_t>(dst) |
                         static_cast<uintptr_t>(row_bytes);
  if ((alignment & 15) == 0) {
    const int64_t num_vec = row_bytes / 16;
    const copy_unit16* src4 = reinterpret_cast<const copy_unit16*>(src);
    copy_unit16* dst4 = reinterpret_cast<copy_unit16*>(dst);
    for (int64_t j = lane_id; j < num_vec; j += sg_size) {
      dst4[j] = src4[j];
    }
    return;
  }

  if ((alignment & 3) == 0) {
    const int64_t num_words = row_bytes / 4;
    const unsigned int* src_words = reinterpret_cast<const unsigned int*>(src);
    unsigned int* dst_words = reinterpret_cast<unsigned int*>(dst);
    for (int64_t j = lane_id; j < num_words; j += sg_size) {
      dst_words[j] = src_words[j];
    }
    return;
  }

  for (int64_t j = lane_id; j < row_bytes; j += sg_size) {
    dst[j] = src[j];
  }
}

// Zero row_bytes at dst with one sub-group, using the same 16/4/1-byte
// alignment choice as copy_row_warp.
inline void zero_row_warp(int lane_id, int sg_size, char* dst,
                          int64_t row_bytes) {
  const auto alignment =
      reinterpret_cast<uintptr_t>(dst) | static_cast<uintptr_t>(row_bytes);
  if ((alignment & 15) == 0) {
    const int64_t num_vec = row_bytes / 16;
    copy_unit16* dst4 = reinterpret_cast<copy_unit16*>(dst);
    for (int64_t j = lane_id; j < num_vec; j += sg_size) {
      dst4[j] = copy_unit16{0u, 0u, 0u, 0u};
    }
    return;
  }

  if ((alignment & 3) == 0) {
    const int64_t num_words = row_bytes / 4;
    unsigned int* dst_words = reinterpret_cast<unsigned int*>(dst);
    for (int64_t j = lane_id; j < num_words; j += sg_size) {
      dst_words[j] = 0u;
    }
    return;
  }

  for (int64_t j = lane_id; j < row_bytes; j += sg_size) {
    dst[j] = 0;
  }
}

// Byte address of row in a paged cache: page (row / block_size) starts at
// block_stride, and the row sits at (row % block_size) * row_bytes inside it.
// No bounds check.
inline const char* cache_row_ptr(const char* cache, int64_t row,
                                 int32_t block_size, int64_t block_stride,
                                 int64_t row_bytes) {
  return cache + (row / block_size) * block_stride +
         (row % block_size) * row_bytes;
}

// Mutable overload of cache_row_ptr. Same address, no bounds check.
inline char* cache_row_ptr(char* cache, int64_t row, int32_t block_size,
                           int64_t block_stride, int64_t row_bytes) {
  return cache + (row / block_size) * block_stride +
         (row % block_size) * row_bytes;
}

// Copy one cache row with one sub-group. value_bytes <= 0 copies row_bytes
// as one contiguous span. Otherwise the value span and the following scale
// span (row_bytes - value_bytes) are copied separately; each is packed by
// offset within its page, so the two spans are not contiguous with each
// other.
inline void copy_cache_row_warp(int lane_id, int sg_size,
                                const char* src_cache, int64_t src_row,
                                int32_t src_block_size,
                                int64_t src_block_stride, char* dst_cache,
                                int64_t dst_row, int32_t dst_block_size,
                                int64_t dst_block_stride, int64_t row_bytes,
                                int64_t value_bytes) {
  if (value_bytes <= 0) {
    copy_row_warp(lane_id, sg_size,
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
  copy_row_warp(lane_id, sg_size, src_page + src_offset * value_bytes,
                dst_page + dst_offset * value_bytes, value_bytes);
  copy_row_warp(
      lane_id, sg_size,
      src_page + src_block_size * value_bytes + src_offset * scale_bytes,
      dst_page + dst_block_size * value_bytes + dst_offset * scale_bytes,
      scale_bytes);
}

// Zero one cache row with one sub-group. value_bytes <= 0 zeros row_bytes
// as one span. Otherwise the value span and the scale span are zeroed
// separately, with the same page packing as copy_cache_row_warp.
inline void zero_cache_row_warp(int lane_id, int sg_size, char* cache,
                                int64_t row, int32_t block_size,
                                int64_t block_stride, int64_t row_bytes,
                                int64_t value_bytes) {
  if (value_bytes <= 0) {
    zero_row_warp(
        lane_id, sg_size,
        cache_row_ptr(cache, row, block_size, block_stride, row_bytes),
        row_bytes);
    return;
  }
  const int64_t scale_bytes = row_bytes - value_bytes;
  const int64_t offset = row % block_size;
  char* page = cache + (row / block_size) * block_stride;
  zero_row_warp(lane_id, sg_size, page + offset * value_bytes, value_bytes);
  zero_row_warp(lane_id, sg_size,
                page + block_size * value_bytes + offset * scale_bytes,
                scale_bytes);
}

// In-place inclusive scan, run by every work-item of sub-group sg, over the
// n entries s_data[offset, offset + n) that are also below count.
// accumulator is added to every entry and the new running total is returned
// to every work-item. n is the number of sub-groups in the work-group, so
// the scan never reads entries this iteration did not write, whether n is
// above or below the sub-group width; it loops over width-sized pieces when
// n is larger.
inline int warp_inclusive_scan(const sycl::sub_group& sg, int32_t* s_data,
                               int lane_id, int sg_size, int offset, int n,
                               int count, int accumulator) {
  for (int base = 0; base < n; base += sg_size) {
    const int idx = offset + base + lane_id;
    const bool in_range = base + lane_id < n && idx < count;
    int val = in_range ? s_data[idx] : 0;
    val = sycl::inclusive_scan_over_group(sg, val, sycl::plus<int>());
    val += accumulator;
    if (in_range) s_data[idx] = val;
    accumulator = sycl::group_broadcast(sg, val, sg_size - 1);
  }
  return accumulator;
}

// Hot-cache row for slot. The block id is hot_block_table[row * table_stride
// + slot / block_size], and the row is that id times block_size plus
// slot % block_size. No bounds check.
inline int64_t get_physical_hot_row(const int32_t* hot_block_table,
                                    int32_t row, int64_t table_stride,
                                    int32_t block_size, int32_t slot) {
  const int32_t block =
      hot_block_table[static_cast<int64_t>(row) * table_stride +
                      slot / block_size];
  return static_cast<int64_t>(block) * block_size + slot % block_size;
}

// Write physical_row to hot_indices[index]. When attention_indices is
// non-null, also write -1 if physical_row < 0, otherwise
// (physical_row / hot_block_size) * attention_block_stride +
// physical_row % hot_block_size. The product is computed in int64.
inline void store_hot_index(int32_t* hot_indices, int32_t* attention_indices,
                            int32_t index, int32_t physical_row,
                            int32_t hot_block_size,
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

// One 1024-work-item work-group per batch row. Translates that row's top-k
// token ids through the optional request, source, and resident tables, then
// assigns a hot slot to every id that still needs one. Resident rows and ids
// that are negative or past host_rows are finished immediately. The rest are
// matched against the LRU list: a hit keeps its slot, a miss takes the
// oldest evictable slot and records the new owner in device_global_indices.
// An LRU slot outside [0, region_stride) is not a hit and is not written.
// The list is rewritten as leftover evictable slots, then the new misses,
// then the hits. A negative state row writes -1 into its hot indexes and,
// when the pointers are non-null, its attention and resolved indexes, sets
// the valid and swap counts to 0 when those pointers are non-null, and
// returns. Requires hot_size >= top_k. Valid global ids must be unique
// within the row. a.hash_size must be 2 * top_k.
//
// Local memory layout (int32 region followed by int16 region), with nbc the
// number of sub-group-wide chunks of hot_size:
//   s_topk[top_k]            top-k global ids; reused as miss scratch
//   s_chunk_off[nbc + 1]     prefix sums for hit (then miss) compaction
//   s_evict_off[nbc + 1]     prefix sums for evictable compaction
//   s_hash_keys[hash_size]   open addressing: global id -> top-k index
//   s_counters[3]            hits, phase-1 resolved, valid count
//   s_lru_out[hot_size]      int16, compacted slots: [hits fwd | evict bwd]
//   s_hash_vals[hash_size]   int16 hash values (top-k index)
void hisparse_resolve_residency_kernel(const HisparseResolveArgs& a,
                                       int32_t hash_size,
                                       const sycl::nd_item<1>& item,
                                       int32_t* smem_raw) {
  const auto group = item.get_group();
  const sycl::sub_group sg = item.get_sub_group();
  const int sg_size = static_cast<int>(sg.get_max_local_range()[0]);
  const int NUM_WARPS = static_cast<int>(sg.get_group_linear_range());
  const int top_k = a.top_k;
  const int hot_size = a.hot_size;
  const int64_t region_stride = a.region_stride;
  const int num_buffer_chunks = (hot_size + sg_size - 1) / sg_size;
  const int num_token_chunks = (top_k + sg_size - 1) / sg_size;
  const int block_dim = static_cast<int>(item.get_local_range(0));

  const int batch_row = static_cast<int>(item.get_group(0));
  const int request_row =
      a.request_ids != nullptr ? a.request_ids[batch_row] : batch_row;
  const int state_row =
      a.request_state_indices != nullptr && request_row >= 0 &&
              request_row < a.request_state_count
          ? a.request_state_indices[request_row]
          : (a.request_state_indices == nullptr ? request_row : -1);
  const int tid = static_cast<int>(item.get_local_linear_id());
  // V2 publishes -1 for graph-capture padding rows.
  if (state_row < 0) {
    for (int i = tid; i < top_k; i += block_dim) {
      const int64_t index = static_cast<int64_t>(batch_row) * top_k + i;
      a.hot_indices[index] = -1;
      if (a.attention_indices != nullptr) {
        a.attention_indices[static_cast<int64_t>(batch_row) *
                                a.attention_row_stride +
                            i] = -1;
      }
      if (a.resolved_global_indices != nullptr) {
        a.resolved_global_indices[index] = -1;
      }
    }
    if (a.valid_counts != nullptr && tid == 0) {
      a.valid_counts[static_cast<int64_t>(batch_row) * a.valid_count_stride] =
          0;
    }
    if (a.swap_counts != nullptr && tid == 0) {
      a.swap_counts[batch_row] = 0;
    }
    return;
  }
  const int warp_id = static_cast<int>(sg.get_group_linear_id());
  const int lane_id = static_cast<int>(sg.get_local_linear_id());

  const int32_t* row_topk =
      a.global_indices + static_cast<int64_t>(batch_row) * a.input_row_stride;
  int32_t* row_out = a.hot_indices + static_cast<int64_t>(batch_row) * top_k;
  int32_t* row_attention =
      a.attention_indices != nullptr
          ? a.attention_indices +
                static_cast<int64_t>(batch_row) * a.attention_row_stride
          : nullptr;
  int32_t* row_miss =
      (a.miss_mask != nullptr)
          ? a.miss_mask + static_cast<int64_t>(batch_row) * top_k
          : nullptr;
  int32_t* row_dgi =
      a.device_global_indices + static_cast<int64_t>(state_row) * region_stride;
  int16_t* row_lru = a.lru_slots + static_cast<int64_t>(state_row) * hot_size;

  int32_t* s_topk = smem_raw;
  int32_t* s_chunk_off = s_topk + top_k;
  int32_t* s_evict_off = s_chunk_off + (num_buffer_chunks + 1);
  int32_t* s_hash_keys = s_evict_off + (num_buffer_chunks + 1);
  int32_t* s_counters = s_hash_keys + hash_size;
  int16_t* s_lru_out = reinterpret_cast<int16_t*>(s_counters + 3);
  int16_t* s_hash_vals = s_lru_out + hot_size;

  if (tid < 3) {
    s_counters[tid] = 0;
  }
  sycl::group_barrier(group);

  // Phase 1: translate request-relative positions and resolve resident rows.
  for (int i = tid; i < top_k; i += block_dim) {
    const int32_t token_index = row_topk[i];
    int32_t g = token_index;
    int32_t resident_row = -1;
    if (a.source_block_table != nullptr) {
      const int32_t request_id = a.request_ids[batch_row];
      const int32_t source_block =
          token_index >= 0 ? token_index / a.source_block_size : -1;
      if (request_id >= 0 && request_id < a.source_num_reqs &&
          source_block >= 0 && source_block < a.source_num_blocks) {
        const int32_t physical_block =
            a.source_block_table[static_cast<int64_t>(request_id) *
                                     a.source_bt_stride +
                                 source_block];
        g = physical_block > 0 ? physical_block * a.source_block_size +
                                     token_index % a.source_block_size
                               : -1;
      } else {
        g = -1;
      }
      if (a.resident_block_table != nullptr) {
        const int32_t resident_block =
            token_index >= 0 ? token_index / a.resident_block_size : -1;
        if (request_id >= 0 && request_id < a.resident_num_reqs &&
            resident_block >= 0 && resident_block < a.resident_num_blocks) {
          const int32_t physical_block =
              a.resident_block_table[static_cast<int64_t>(request_id) *
                                         a.resident_bt_stride +
                                     resident_block];
          if (physical_block != a.resident_null_block && physical_block >= 0) {
            resident_row = physical_block * a.resident_block_size +
                           token_index % a.resident_block_size;
          }
        }
      }
    }
    if (g >= a.host_rows) {
      g = -1;
    }
    if (a.resolved_global_indices != nullptr) {
      a.resolved_global_indices[static_cast<int64_t>(batch_row) * top_k + i] =
          g;
    }
    if (resident_row >= 0 || g >= 0) local_atomic_int(s_counters[2]).fetch_add(1);
    if (row_miss != nullptr) row_miss[i] = 0;
    if (resident_row >= 0) {
      store_hot_index(row_out, row_attention, i, resident_row,
                      a.hot_block_size, a.attention_block_stride);
      s_topk[i] = kTokenDone;
      local_atomic_int(s_counters[1]).fetch_add(1);
    } else if (g < 0) {
      store_hot_index(row_out, row_attention, i, -1, a.hot_block_size,
                      a.attention_block_stride);
      s_topk[i] = kTokenDone;
      local_atomic_int(s_counters[1]).fetch_add(1);
    } else {
      s_topk[i] = g;
    }
  }
  sycl::group_barrier(group);
  if (a.valid_counts != nullptr && tid == 0) {
    a.valid_counts[static_cast<int64_t>(batch_row) * a.valid_count_stride] =
        s_counters[2];
  }
  // Fully resident rows need only request-relative page translation. Avoid
  // scanning or rewriting the hot LRU when no selected row can consult it.
  if (s_counters[1] == top_k) {
    if (tid == 0 && a.swap_counts != nullptr) {
      a.swap_counts[batch_row] = 0;
    }
    return;
  }

  for (int i = tid; i < hash_size; i += block_dim) {
    s_hash_keys[i] = kHashEmpty;
  }
  for (int i = tid; i < num_buffer_chunks + 1; i += block_dim) {
    s_chunk_off[i] = 0;
    s_evict_off[i] = 0;
  }
  sycl::group_barrier(group);

  for (int i = tid; i < top_k; i += block_dim) {
    const int32_t g = s_topk[i];
    if (g == kTokenDone) {
      continue;
    }
    int slot = hash_slot(g, hash_size);
    while (true) {
      int32_t old = kHashEmpty;
      local_atomic_int(s_hash_keys[slot]).compare_exchange_strong(old, g);
      if (old == kHashEmpty || old == g) {
        s_hash_vals[slot] = static_cast<int16_t>(i);
        break;
      }
      slot = (slot + 1) % hash_size;
    }
  }
  sycl::group_barrier(group);

  // Phase 2: walk hot slots in LRU order, classify hit / evictable, and
  // compact them (hits forward, evictables backward) into s_lru_out.
  const int iters_buffer = (num_buffer_chunks + NUM_WARPS - 1) / NUM_WARPS;
  int total_hit_count = 0;
  int total_evict_count = 0;
  for (int iter = 0; iter < iters_buffer; iter++) {
    const int chunk_idx = warp_id + iter * NUM_WARPS;
    const bool has_valid_chunk = chunk_idx < num_buffer_chunks;

    const int pos = chunk_idx * sg_size + lane_id;
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
                          a.hot_block_table, request_row, a.hot_table_stride,
                          a.hot_block_size, slot)),
                      a.hot_block_size, a.attention_block_stride);
    }

    int local_hit_off = 0;
    int local_evict_off = 0;
    if (has_valid_chunk) {
      const int hit = is_hit ? 1 : 0;
      const int evict = is_evictable ? 1 : 0;
      local_hit_off =
          sycl::exclusive_scan_over_group(sg, hit, sycl::plus<int>());
      local_evict_off =
          sycl::exclusive_scan_over_group(sg, evict, sycl::plus<int>());
      const int hit_count = sycl::reduce_over_group(sg, hit, sycl::plus<int>());
      const int evict_count =
          sycl::reduce_over_group(sg, evict, sycl::plus<int>());
      if (lane_id == 0) {
        s_chunk_off[chunk_idx + 1] = hit_count;
        s_evict_off[chunk_idx + 1] = evict_count;
      }
    }
    sycl::group_barrier(group);

    if (warp_id == 0) {
      total_hit_count = warp_inclusive_scan(
          sg, s_chunk_off, lane_id, sg_size, chunk_idx + 1, NUM_WARPS,
          num_buffer_chunks + 1, total_hit_count);
      total_evict_count = warp_inclusive_scan(
          sg, s_evict_off, lane_id, sg_size, chunk_idx + 1, NUM_WARPS,
          num_buffer_chunks + 1, total_evict_count);
      if (tid == 0) {
        s_counters[0] = total_hit_count;
      }
    }
    sycl::group_barrier(group);

    if (is_hit) {
      const int off = s_chunk_off[chunk_idx] + local_hit_off;
      s_lru_out[off] = slot;
    }
    if (is_evictable) {
      const int off = s_evict_off[chunk_idx] + local_evict_off;
      s_lru_out[hot_size - 1 - off] = slot;
    }
  }
  sycl::group_barrier(group);

  // Reset prefix sums for the miss compaction (token chunks <= buffer
  // chunks because hot_size >= top_k).
  for (int i = tid; i < num_token_chunks + 1; i += block_dim) {
    s_chunk_off[i] = 0;
  }
  sycl::group_barrier(group);

  // Phase 3: compact misses, assign them eviction slots (oldest first) and
  // record the new ownership in device_global_indices.
  const int iters_token = (num_token_chunks + NUM_WARPS - 1) / NUM_WARPS;
  int miss_running_total = 0;
  for (int iter = 0; iter < iters_token; iter++) {
    const int chunk_idx = warp_id + iter * NUM_WARPS;
    const bool has_valid_chunk = chunk_idx < num_token_chunks;

    const int i = chunk_idx * sg_size + lane_id;
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
      const int miss = is_miss ? 1 : 0;
      local_miss_off =
          sycl::exclusive_scan_over_group(sg, miss, sycl::plus<int>());
      const int miss_count =
          sycl::reduce_over_group(sg, miss, sycl::plus<int>());
      if (lane_id == 0) {
        s_chunk_off[chunk_idx + 1] = miss_count;
      }
    }
    sycl::group_barrier(group);

    if (warp_id == 0) {
      miss_running_total = warp_inclusive_scan(
          sg, s_chunk_off, lane_id, sg_size, chunk_idx + 1, NUM_WARPS,
          num_token_chunks + 1, miss_running_total);
    }
    sycl::group_barrier(group);

    if (is_miss) {
      const int m = s_chunk_off[chunk_idx] + local_miss_off;
      const int16_t evict_slot = s_lru_out[hot_size - 1 - m];
      if (evict_slot < 0 || evict_slot >= region_stride) {
        // Corruption tripwire: an out-of-range slot (corrupted lru state,
        // see phase 2) must not become a hot-cache or dgi write. Resolve the
        // entry as invalid (-1, masked by attention, not in the miss set).
        // The compact record keeps the host id and a device slot of -1, and
        // the gather kernels skip that copy.
        store_hot_index(row_out, row_attention, i, -1, a.hot_block_size,
                        a.attention_block_stride);
        if (a.swap_host_physical_rows != nullptr) {
          const int64_t compact_index =
              static_cast<int64_t>(batch_row) * top_k + m;
          a.swap_host_physical_rows[compact_index] = g;
          a.swap_device_physical_rows[compact_index] = -1;
        }
      } else {
        // Reuse s_topk as compacted miss scratch: m < i always holds (done
        // entries are skipped), so writes never overrun pending reads.
        s_topk[m] = g;
        const int32_t physical_row = static_cast<int32_t>(get_physical_hot_row(
            a.hot_block_table, request_row, a.hot_table_stride,
            a.hot_block_size, evict_slot));
        store_hot_index(row_out, row_attention, i, physical_row,
                        a.hot_block_size, a.attention_block_stride);
        if (a.swap_host_physical_rows != nullptr) {
          const int64_t compact_index =
              static_cast<int64_t>(batch_row) * top_k + m;
          a.swap_host_physical_rows[compact_index] = g;
          a.swap_device_physical_rows[compact_index] = physical_row;
        }
        if (row_miss != nullptr) row_miss[i] = 1;
        row_dgi[evict_slot] = g;
      }
    }
  }
  sycl::group_barrier(group);

  const int total_hits = s_counters[0];
  const int total_misses = top_k - total_hits - s_counters[1];
  if (a.swap_counts != nullptr && tid == 0) {
    a.swap_counts[batch_row] = total_misses;
  }
  if (a.stats != nullptr && tid == 0) {
    global_atomic_u64(a.stats[0])
        .fetch_add(static_cast<unsigned long long>(total_hits));
    global_atomic_u64(a.stats[1])
        .fetch_add(static_cast<unsigned long long>(total_misses));
  }

  // Phase 4: write back the LRU order: stale evictables at the front,
  // freshly loaded misses next, then hits at MRU.
  const int total_evictable = hot_size - total_hits;
  const int remaining_evictable = total_evictable - total_misses;
  for (int i = tid; i < hot_size; i += block_dim) {
    if (i < remaining_evictable) {
      row_lru[i] = s_lru_out[hot_size - 1 - total_misses - i];
    } else if (i < remaining_evictable + total_misses) {
      row_lru[i] = s_lru_out[hot_size - 1 - (i - remaining_evictable)];
    } else {
      row_lru[i] = s_lru_out[i - remaining_evictable - total_misses];
    }
  }
}

}  // namespace

// Largest entry of the device's info::device::sub_group_sizes, or 1 when
// the device lists none.
int32_t hisparse_max_sub_group_size(const sycl::queue& q) {
  const std::vector<size_t> sizes =
      q.get_device().get_info<sycl::info::device::sub_group_sizes>();
  if (sizes.empty()) return 1;
  return static_cast<int32_t>(*std::max_element(sizes.begin(), sizes.end()));
}

// Run body with kSubGroupSize bound to sg, when sg is 4, 8, 16, 32, or 64.
// body is a statement that submits a kernel marked
// [[sycl::reqd_sub_group_size(kSubGroupSize)]]. When sg is anything else,
// print it and return 1 from the enclosing function; body does not run.
#define HISPARSE_WITH_SUB_GROUP_SIZE(sg, body)                               \
  switch (sg) {                                                              \
    case 64: {                                                               \
      constexpr int kSubGroupSize = 64;                                      \
      body;                                                                  \
    } break;                                                                 \
    case 32: {                                                               \
      constexpr int kSubGroupSize = 32;                                      \
      body;                                                                  \
    } break;                                                                 \
    case 16: {                                                               \
      constexpr int kSubGroupSize = 16;                                      \
      body;                                                                  \
    } break;                                                                 \
    case 8: {                                                                \
      constexpr int kSubGroupSize = 8;                                       \
      body;                                                                  \
    } break;                                                                 \
    case 4: {                                                                \
      constexpr int kSubGroupSize = 4;                                       \
      body;                                                                  \
    } break;                                                                 \
    default:                                                                 \
      fprintf(stderr,                                                        \
              "HiSparse sub-group size %d is not 4, 8, 16, 32, or 64\n",     \
              static_cast<int>(sg));                                         \
      return 1;                                                              \
  }

// Local memory for one resolve work-group when hot_size >= top_k and the
// sub-group size is at least sub_group_size. int32 region: top_k ids, two
// prefix arrays of (ceil(hot_size/sub_group_size)+1), a hash table of
// 2*top_k keys, and 3 counters. int16 region: hot_size compacted slots and
// 2*top_k hash values. Does not check the arguments.
size_t hisparse_resolve_smem_bytes(int32_t top_k, int32_t hot_size,
                                   int32_t sub_group_size) {
  const int hash_size = 2 * top_k;
  const int num_buffer_chunks =
      (hot_size + sub_group_size - 1) / sub_group_size;
  return sizeof(int32_t) * (static_cast<size_t>(top_k) +
                            2 * (static_cast<size_t>(num_buffer_chunks) + 1) +
                            static_cast<size_t>(hash_size) + 3) +
         sizeof(int16_t) *
             (static_cast<size_t>(hot_size) + static_cast<size_t>(hash_size));
}

// Submit one 1024-work-item resolve work-group per batch row. Return 0
// without submitting when num_rows or top_k is 0. Return 1, after printing
// the reason, when a source block table is set without request_ids,
// hot_size < top_k, hot_size > 32768, the device's largest sub-group size is
// not 4, 8, 16, 32, or 64, or the local memory, sized with that sub-group
// size and rounded up to whole int32 words, exceeds the device's
// local_mem_size.
int hisparse_resolve_residency_launch(const HisparseResolveArgs& args,
                                      sycl::queue& q) {
  if (args.num_rows == 0 || args.top_k == 0) return 0;
  if (args.source_block_table != nullptr && args.request_ids == nullptr) {
    fprintf(stderr, "HiSparse source_block_table requires request_ids\n");
    return 1;
  }
  if (args.hot_size < args.top_k) {
    fprintf(stderr, "HiSparse requires hot_size >= top_k (%d < %d)\n",
            args.hot_size, args.top_k);
    return 1;
  }
  // LRU slots and hash values are stored as int16.
  if (args.hot_size > std::numeric_limits<int16_t>::max() + 1) {
    fprintf(stderr, "HiSparse hot_size %d exceeds the int16 slot range\n",
            args.hot_size);
    return 1;
  }
  constexpr size_t kBlockSize = 1024;
  const int32_t sub_group_size = hisparse_max_sub_group_size(q);
  const int32_t hash_size = 2 * args.top_k;
  const size_t smem_bytes = hisparse_resolve_smem_bytes(
      args.top_k, args.hot_size, sub_group_size);
  const size_t smem_words = (smem_bytes + sizeof(int32_t) - 1) / sizeof(int32_t);
  const size_t local_mem =
      q.get_device().get_info<sycl::info::device::local_mem_size>();
  if (smem_words * sizeof(int32_t) > local_mem) {
    fprintf(stderr, "HiSparse local memory %zu exceeds the device's %zu\n",
            smem_words * sizeof(int32_t), local_mem);
    return 1;
  }
  const HisparseResolveArgs a = args;
  HISPARSE_WITH_SUB_GROUP_SIZE(
      sub_group_size,
      q.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<int32_t, 1> smem(sycl::range<1>(smem_words), cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(args.num_rows * kBlockSize),
                              sycl::range<1>(kBlockSize)),
            [=](sycl::nd_item<1> item)
                [[sycl::reqd_sub_group_size(kSubGroupSize)]] {
                  hisparse_resolve_residency_kernel(
                      a, hash_size, item,
                      smem.get_multi_ptr<sycl::access::decorated::no>().get());
                });
      }););
  return 0;
}

// Submit gather_plan as a (num_rows, num_chunks) grid of 1024-work-item
// work-groups. With W sub-groups per work-group (1024 divided by the
// device's largest sub-group size), num_chunks is
// min(ceil(top_k / W), max(1, 256 / num_rows)); columns beyond that are still
// visited by the kernel's stride. Return 0 without submitting when num_rows
// or top_k is 0. Return 1 when the largest sub-group size is not 4, 8, 16,
// 32, or 64, or does not divide 1024.
//
// Kernel: the first group index is the batch row, and columns are striped
// across the second group index and the sub-groups. A row whose
// request_state_indices entry is negative is padding: attention is set to -1
// and nothing is copied. Every other column gets an attention id from its
// hot index. A column is copied only when its miss flag is nonzero, both
// indexes are in range, and the host index is below host_rows; a host index
// at or past host_rows zeros the hot slot instead. miss_mask is required.
int hisparse_gather_plan_launch(
    const char* host_cache, char* hot_cache, const int32_t* global_indices,
    const int32_t* hot_indices, const int32_t* miss_mask,
    int32_t* attention_indices, const int32_t* request_state_indices,
    int64_t host_rows, int32_t host_block_size, int64_t host_block_stride,
    int64_t hot_rows, int64_t row_bytes, int64_t row_value_bytes,
    int64_t hot_block_stride, int32_t hot_block_size, int32_t num_rows,
    int32_t top_k, int64_t attention_block_stride, sycl::queue& q) {
  if (num_rows == 0 || top_k == 0) return 0;
  constexpr int kBlockSize = 1024;
  const int32_t sub_group_size = hisparse_max_sub_group_size(q);
  if (sub_group_size <= 0 || kBlockSize % sub_group_size != 0) {
    fprintf(stderr,
            "HiSparse sub-group size %d does not divide the %d-wide work-group\n",
            sub_group_size, kBlockSize);
    return 1;
  }
  const int kNumWarps = kBlockSize / sub_group_size;
  constexpr int kTargetBlocks = 256;
  const int max_chunks = std::max(1, (top_k + kNumWarps - 1) / kNumWarps);
  const int num_chunks =
      std::min(max_chunks, std::max(1, kTargetBlocks / num_rows));
  HISPARSE_WITH_SUB_GROUP_SIZE(
      sub_group_size,
      q.parallel_for(
          sycl::nd_range<2>(sycl::range<2>(num_rows, num_chunks * kBlockSize),
                            sycl::range<2>(1, kBlockSize)),
          [=](sycl::nd_item<2> item)
              [[sycl::reqd_sub_group_size(kSubGroupSize)]] {
        const sycl::sub_group sg = item.get_sub_group();
        const int sg_size = static_cast<int>(sg.get_max_local_range()[0]);
        const int NUM_WARPS = static_cast<int>(sg.get_group_linear_range());
        // Columns are interleaved across the second grid dimension so
        // few-row launches (local-prefill staging's single-row layout, small
        // decode batches) still fill the device with outstanding host reads
        // instead of starving on one work-group per row.
        const int row = static_cast<int>(item.get_group(0));
        const bool is_padding =
            request_state_indices != nullptr && request_state_indices[row] < 0;
        const int warp_id = static_cast<int>(sg.get_group_linear_id());
        const int lane_id = static_cast<int>(sg.get_local_linear_id());
        const int col_start =
            static_cast<int>(item.get_group(1)) * NUM_WARPS + warp_id;
        const int col_stride =
            static_cast<int>(item.get_group_range(1)) * NUM_WARPS;
        const int64_t base = static_cast<int64_t>(row) * top_k;
        for (int col = col_start; col < top_k; col += col_stride) {
          const int32_t dst = hot_indices[base + col];
          if (lane_id == 0 && attention_indices != nullptr) {
            attention_indices[base + col] =
                is_padding
                    ? -1
                    : (dst < 0 ? -1
                               : (dst / hot_block_size) *
                                         attention_block_stride +
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
            copy_cache_row_warp(lane_id, sg_size, host_cache, g,
                                host_block_size, host_block_stride, hot_cache,
                                dst, hot_block_size, hot_block_stride,
                                row_bytes, row_value_bytes);
          } else {
            // No source row for g: zero the planned slot rather than serving
            // whatever bytes it held.
            zero_cache_row_warp(lane_id, sg_size, hot_cache, dst,
                                hot_block_size, hot_block_stride, row_bytes,
                                row_value_bytes);
          }
        }
      }););
  return 0;
}

// Submit gather_compact as a (num_rows, min(8, max(1, 64 / num_rows))) grid
// of 512-work-item work-groups. Return 0 without submitting when num_rows or
// top_k is 0. Return 1 when the largest sub-group size is not 4, 8, 16, 32,
// or 64.
//
// Kernel: for each batch row the miss count is clamped to [0, top_k], and
// those leading compact entries are striped across the second group index
// and the sub-groups. An entry is skipped when the host index is negative or
// the hot index is out of range. A host index below host_rows is copied; one
// at or past host_rows zeros the hot slot.
int hisparse_gather_compact_launch(
    const char* host_cache, char* hot_cache, const int32_t* miss_global_indices,
    const int32_t* miss_hot_indices, const int32_t* miss_counts,
    int64_t host_rows, int32_t host_block_size, int64_t host_block_stride,
    int64_t hot_rows, int64_t row_bytes, int64_t row_value_bytes,
    int64_t hot_block_stride, int32_t hot_block_size, int32_t num_rows,
    int32_t top_k, sycl::queue& q) {
  if (num_rows == 0 || top_k == 0) return 0;
  constexpr int kBlockSize = 512;
  constexpr int kTargetBlocks = 64;
  const int num_chunks = std::min(8, std::max(1, kTargetBlocks / num_rows));
  const int32_t sub_group_size = hisparse_max_sub_group_size(q);
  HISPARSE_WITH_SUB_GROUP_SIZE(
      sub_group_size,
      q.parallel_for(
          sycl::nd_range<2>(sycl::range<2>(num_rows, num_chunks * kBlockSize),
                            sycl::range<2>(1, kBlockSize)),
          [=](sycl::nd_item<2> item)
              [[sycl::reqd_sub_group_size(kSubGroupSize)]] {
        const sycl::sub_group sg = item.get_sub_group();
        const int sg_size = static_cast<int>(sg.get_max_local_range()[0]);
        const int NUM_WARPS = static_cast<int>(sg.get_group_linear_range());
        const int row = static_cast<int>(item.get_group(0));
        const int warp_id = static_cast<int>(sg.get_group_linear_id());
        const int lane_id = static_cast<int>(sg.get_local_linear_id());
        const int miss_count = sycl::min(sycl::max(miss_counts[row], 0), top_k);
        const int col_start =
            static_cast<int>(item.get_group(1)) * NUM_WARPS + warp_id;
        const int col_stride =
            static_cast<int>(item.get_group_range(1)) * NUM_WARPS;
        const int64_t base = static_cast<int64_t>(row) * top_k;
        for (int col = col_start; col < miss_count; col += col_stride) {
          const int32_t g = miss_global_indices[base + col];
          const int32_t dst = miss_hot_indices[base + col];
          if (g < 0 || dst < 0 || dst >= hot_rows) {
            continue;
          }
          if (g < host_rows) {
            copy_cache_row_warp(lane_id, sg_size, host_cache, g,
                                host_block_size, host_block_stride, hot_cache,
                                dst, hot_block_size, hot_block_stride,
                                row_bytes, row_value_bytes);
          } else {
            zero_cache_row_warp(lane_id, sg_size, hot_cache, dst,
                                hot_block_size, hot_block_stride, row_bytes,
                                row_value_bytes);
          }
        }
      }););
  return 0;
}

// Submit backup with 256-work-item work-groups, rounding the grid up so each
// item has a sub-group of the device's largest size.
// Return 0 without submitting when num_items is 0. Return 1 when that size
// is not 4, 8, 16, 32, or 64, or does not divide 256.
//
// Kernel: one sub-group per item, grid-stride over num_items. Copy
// src_cache[src_indices[i]] into host_cache[dst_slots[i]]. Skip the item when
// the source is outside [0, src_rows) or the destination is outside
// [0, host_rows).
int hisparse_backup_launch(
    const char* src_cache, const int64_t* src_indices, char* host_cache,
    const int64_t* dst_slots, int64_t row_bytes, int64_t row_value_bytes,
    int64_t src_block_stride, int32_t src_block_size,
    int64_t host_block_stride, int32_t host_block_size, int32_t num_items,
    int64_t src_rows, int64_t host_rows, sycl::queue& q) {
  if (num_items == 0) return 0;
  constexpr int kBlockSize = 256;
  const int32_t sub_group_size = hisparse_max_sub_group_size(q);
  if (sub_group_size <= 0 || kBlockSize % sub_group_size != 0) {
    fprintf(stderr,
            "HiSparse sub-group size %d does not divide the %d-wide work-group\n",
            sub_group_size, kBlockSize);
    return 1;
  }
  const int kNumWarps = kBlockSize / sub_group_size;
  const int grid = (num_items + kNumWarps - 1) / kNumWarps;
  HISPARSE_WITH_SUB_GROUP_SIZE(
      sub_group_size,
      q.parallel_for(
          sycl::nd_range<1>(
              sycl::range<1>(static_cast<size_t>(grid) * kBlockSize),
              sycl::range<1>(kBlockSize)),
          [=](sycl::nd_item<1> item)
              [[sycl::reqd_sub_group_size(kSubGroupSize)]] {
        const sycl::sub_group sg = item.get_sub_group();
        const int sg_size = static_cast<int>(sg.get_max_local_range()[0]);
        const int NUM_WARPS = static_cast<int>(sg.get_group_linear_range());
        const int lane_id = static_cast<int>(sg.get_local_linear_id());
        const int warp_id = static_cast<int>(item.get_group(0)) * NUM_WARPS +
                            static_cast<int>(sg.get_group_linear_id());
        const int total_warps =
            static_cast<int>(item.get_group_range(0)) * NUM_WARPS;
        for (int i = warp_id; i < num_items; i += total_warps) {
          const int64_t s = src_indices[i];
          const int64_t d = dst_slots[i];
          if (s < 0 || s >= src_rows || d < 0 || d >= host_rows) {
            continue;
          }
          copy_cache_row_warp(lane_id, sg_size, src_cache, s, src_block_size,
                              src_block_stride, host_cache, d,
                              host_block_size, host_block_stride, row_bytes,
                              row_value_bytes);
        }
      }););
  return 0;
}

// Submit one 256-work-item work-group per token. Return 0 without submitting
// when num_tokens, num_state_rows, or region_stride is 0.
//
// Kernel: look up the token's request and state row, then set every
// device_global_indices entry in the state row that equals the written slot
// to -1. Do nothing when the request is outside [0, num_request_ids), the
// state is outside [0, num_state_rows), or the written slot is negative.
int hisparse_invalidate_written_slots_launch(
    int32_t* device_global_indices, const int32_t* request_state_indices,
    const int32_t* req_id_per_token, const int64_t* written_slots,
    int64_t num_tokens, int64_t num_request_ids, int64_t num_state_rows,
    int64_t region_stride, sycl::queue& q) {
  if (num_tokens == 0 || num_state_rows == 0 || region_stride == 0) {
    return 0;
  }
  constexpr size_t kBlockSize = 256;
  q.parallel_for(
      sycl::nd_range<1>(
          sycl::range<1>(static_cast<size_t>(num_tokens) * kBlockSize),
          sycl::range<1>(kBlockSize)),
      [=](sycl::nd_item<1> item) {
        const int64_t token_idx = static_cast<int64_t>(item.get_group(0));
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
        for (int64_t offset = static_cast<int64_t>(item.get_local_id(0));
             offset < region_stride;
             offset += static_cast<int64_t>(item.get_local_range(0))) {
          if (row[offset] == written_slot) {
            row[offset] = -1;
          }
        }
      });
  return 0;
}
