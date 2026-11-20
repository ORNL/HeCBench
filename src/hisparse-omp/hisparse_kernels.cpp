// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the SGLang project
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// HiSparse decode hot-buffer kernels for sparse MLA, OpenMP target offload
// port.
//
// There is no warp in this port. The row copies of backup and the two
// gathers give each row a group of 64 consecutive loop iterations that
// stride over its units, where a unit is 16, 4, or 1 bytes depending on
// alignment. Resolve gives each batch row to one team, which runs the same
// phases as the CUDA kernel in order: translate, hash insert, LRU walk with
// hits compacted forward and evictables backward, miss assignment, and LRU
// write-back. Every phase is spread over the team's threads; hash insertion
// claims slots with compare-and-swap, and the two compactions use a team
// prefix sum over contiguous chunks, so the outputs match the CPU reference
// exactly. Invalidation gives each written token to one team.
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

#include <omp.h>

#include <cstdint>
#include <cstdio>
#include <limits>

namespace {

// 16-byte copy unit. A clang vector stays one 128-bit access; a struct copy
// may be split into narrower strided stores, which is slow on host memory.
#if defined(__clang__)
typedef unsigned int copy_unit16 __attribute__((ext_vector_type(4)));
#else
struct alignas(16) copy_unit16 {
  uint32_t x, y, z, w;
};
#endif

#pragma omp declare target

// Sentinel in the top-k scratch: entry already resolved (hit / newest /
// invalid), no miss handling needed.
constexpr int32_t kTokenDone = -1;
constexpr int32_t kHashEmpty = -1;

// First open-addressing slot for key. The multiplier is Knuth's
// 2654435761, the key is taken as an unsigned 32-bit pattern, and the
// result is that product modulo hash_size. hash_size must be positive.
inline int32_t hash_slot(int32_t key, int32_t hash_size) {
  return static_cast<int32_t>((static_cast<uint32_t>(key) * 2654435761u) %
                              static_cast<uint32_t>(hash_size));
}

// Byte offset of byte b of cache row row. value_bytes <= 0 treats the row as
// one span of row_bytes at (row % block_size) * row_bytes inside page
// row / block_size. Otherwise b < value_bytes lies in the value span at
// (row % block_size) * value_bytes, and the rest lies in the scale span that
// follows the page's block_size value spans. No bounds check.
inline int64_t row_byte_offset(int64_t row, int32_t block_size,
                               int64_t block_stride, int64_t row_bytes,
                               int64_t value_bytes, int64_t b) {
  const int64_t page = (row / block_size) * block_stride;
  const int64_t offset = row % block_size;
  if (value_bytes <= 0) return page + offset * row_bytes + b;
  if (b < value_bytes) return page + offset * value_bytes + b;
  return page + block_size * value_bytes + offset * (row_bytes - value_bytes) +
         (b - value_bytes);
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

// Exclusive prefix sum of v across the nthreads threads of the calling
// parallel region, using buf[0, nthreads) as work space. Every thread must
// call it with the same buf and nthreads. Returns the sum of v over threads
// below tid and stores the sum over all threads in total. buf may be reused
// as soon as the call returns.
inline int team_exclusive_scan(int32_t* buf, int v, int tid, int nthreads,
                               int& total) {
  buf[tid] = v;
  #pragma omp barrier
  for (int d = 1; d < nthreads; d <<= 1) {
    const int x = tid >= d ? buf[tid - d] : 0;
    #pragma omp barrier
    buf[tid] += x;
    #pragma omp barrier
  }
  total = buf[nthreads - 1];
  const int exclusive = buf[tid] - v;
  #pragma omp barrier
  return exclusive;
}

// Resolve batch row batch_row on the calling team, with the semantics of
// the CUDA hisparse_resolve_residency_kernel. Translates the row's top-k
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
// within the row. Uses this row's slice of the scratch buffers and
// team_scan[0, nthreads) as prefix-sum work space. Every thread tid of the
// calling parallel region, which has nthreads threads, must call it with
// the same batch_row; it contains barriers. Hits and evictables are
// compacted, and misses numbered, in the same order as the CPU reference.
void resolve_row(const HisparseResolveArgs& a, int batch_row, int tid,
                 int nthreads, int32_t* team_scan) {
  const int top_k = a.top_k;
  const int hot_size = a.hot_size;
  const int hash_size = 2 * top_k;
  const int64_t region_stride = a.region_stride;

  const int request_row =
      a.request_ids != nullptr ? a.request_ids[batch_row] : batch_row;
  const int state_row =
      a.request_state_indices != nullptr && request_row >= 0 &&
              request_row < a.request_state_count
          ? a.request_state_indices[request_row]
          : (a.request_state_indices == nullptr ? request_row : -1);
  // V2 publishes -1 for graph-capture padding rows.
  if (state_row < 0) {
    for (int i = tid; i < top_k; i += nthreads) {
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
    if (tid == 0 && a.valid_counts != nullptr) {
      a.valid_counts[static_cast<int64_t>(batch_row) * a.valid_count_stride] =
          0;
    }
    if (tid == 0 && a.swap_counts != nullptr) {
      a.swap_counts[batch_row] = 0;
    }
    return;
  }

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

  int32_t* s_topk = a.scratch_topk + static_cast<int64_t>(batch_row) * top_k;
  int32_t* s_hash_keys =
      a.scratch_hash_keys + static_cast<int64_t>(batch_row) * hash_size;
  int16_t* s_hash_vals =
      a.scratch_hash_vals + static_cast<int64_t>(batch_row) * hash_size;
  int16_t* s_lru_out =
      a.scratch_lru_out + static_cast<int64_t>(batch_row) * hot_size;
  int16_t* s_found =
      a.scratch_found + static_cast<int64_t>(batch_row) * hot_size;

  // Phase 1: translate request-relative positions and resolve resident rows.
  int local_phase1 = 0;
  int local_valid = 0;
  for (int i = tid; i < top_k; i += nthreads) {
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
    if (resident_row >= 0 || g >= 0) local_valid++;
    if (row_miss != nullptr) row_miss[i] = 0;
    if (resident_row >= 0) {
      store_hot_index(row_out, row_attention, i, resident_row,
                      a.hot_block_size, a.attention_block_stride);
      s_topk[i] = kTokenDone;
      local_phase1++;
    } else if (g < 0) {
      store_hot_index(row_out, row_attention, i, -1, a.hot_block_size,
                      a.attention_block_stride);
      s_topk[i] = kTokenDone;
      local_phase1++;
    } else {
      s_topk[i] = g;
    }
  }
  int phase1 = 0;
  int valid = 0;
  team_exclusive_scan(team_scan, local_phase1, tid, nthreads, phase1);
  team_exclusive_scan(team_scan, local_valid, tid, nthreads, valid);
  if (tid == 0 && a.valid_counts != nullptr) {
    a.valid_counts[static_cast<int64_t>(batch_row) * a.valid_count_stride] =
        valid;
  }
  // Fully resident rows need only request-relative page translation. Avoid
  // scanning or rewriting the hot LRU when no selected row can consult it.
  if (phase1 == top_k) {
    if (tid == 0 && a.swap_counts != nullptr) {
      a.swap_counts[batch_row] = 0;
    }
    return;
  }

  for (int i = tid; i < hash_size; i += nthreads) {
    s_hash_keys[i] = kHashEmpty;
  }
  #pragma omp barrier
  // Claim a slot with compare-and-swap; global ids are unique within the
  // row, so each claimed key's value is written by exactly one thread.
  for (int i = tid; i < top_k; i += nthreads) {
    const int32_t g = s_topk[i];
    if (g == kTokenDone) {
      continue;
    }
    int slot = hash_slot(g, hash_size);
    while (true) {
      int32_t old;
      #pragma omp atomic compare capture
      {
        old = s_hash_keys[slot];
        if (s_hash_keys[slot] == kHashEmpty) {
          s_hash_keys[slot] = g;
        }
      }
      if (old == kHashEmpty || old == g) break;
      slot = (slot + 1) % hash_size;
    }
    s_hash_vals[slot] = static_cast<int16_t>(i);
  }
  #pragma omp barrier

  // Phase 2: look up every hot slot's owner in parallel, then walk the slots
  // in LRU order, classify hit / evictable, and compact them (hits forward,
  // evictables backward) into s_lru_out.
  for (int pos = tid; pos < hot_size; pos += nthreads) {
    const int16_t slot = row_lru[pos];
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
    s_found[pos] = static_cast<int16_t>(found_topk_idx);
  }
  #pragma omp barrier
  // Each thread compacts one contiguous chunk of LRU positions; the prefix
  // sum of per-chunk hit counts gives its first hit and evictable index.
  const int lru_chunk = (hot_size + nthreads - 1) / nthreads;
  const int lru_begin = tid * lru_chunk < hot_size ? tid * lru_chunk : hot_size;
  const int lru_end =
      lru_begin + lru_chunk < hot_size ? lru_begin + lru_chunk : hot_size;
  int chunk_hits = 0;
  for (int pos = lru_begin; pos < lru_end; ++pos) {
    if (s_found[pos] >= 0) chunk_hits++;
  }
  int total_hits = 0;
  int hit_index =
      team_exclusive_scan(team_scan, chunk_hits, tid, nthreads, total_hits);
  int evict_index = lru_begin - hit_index;
  for (int pos = lru_begin; pos < lru_end; ++pos) {
    const int16_t slot = row_lru[pos];
    const int found_topk_idx = s_found[pos];
    if (found_topk_idx >= 0) {
      s_topk[found_topk_idx] = kTokenDone;
      store_hot_index(row_out, row_attention, found_topk_idx,
                      static_cast<int32_t>(get_physical_hot_row(
                          a.hot_block_table, request_row, a.hot_table_stride,
                          a.hot_block_size, slot)),
                      a.hot_block_size, a.attention_block_stride);
      s_lru_out[hit_index++] = slot;
    } else {
      s_lru_out[hot_size - 1 - evict_index++] = slot;
    }
  }
  #pragma omp barrier

  // Phase 3: compact misses, assign them eviction slots (oldest first) and
  // record the new ownership in device_global_indices. Each thread numbers
  // the misses of one contiguous chunk of top-k positions.
  const int topk_chunk = (top_k + nthreads - 1) / nthreads;
  const int topk_begin = tid * topk_chunk < top_k ? tid * topk_chunk : top_k;
  const int topk_end =
      topk_begin + topk_chunk < top_k ? topk_begin + topk_chunk : top_k;
  int chunk_misses = 0;
  for (int i = topk_begin; i < topk_end; ++i) {
    if (s_topk[i] != kTokenDone) chunk_misses++;
  }
  int total_misses = 0;
  int m = team_exclusive_scan(team_scan, chunk_misses, tid, nthreads,
                              total_misses);
  for (int i = topk_begin; i < topk_end; ++i) {
    if (s_topk[i] == kTokenDone) continue;
    const int32_t g = s_topk[i];
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
      const int32_t physical_row = static_cast<int32_t>(get_physical_hot_row(
          a.hot_block_table, request_row, a.hot_table_stride, a.hot_block_size,
          evict_slot));
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
    m++;
  }

  if (tid == 0 && a.swap_counts != nullptr) {
    a.swap_counts[batch_row] = total_misses;
  }
  if (tid == 0 && a.stats != nullptr) {
    #pragma omp atomic update
    a.stats[0] += static_cast<unsigned long long>(total_hits);
    #pragma omp atomic update
    a.stats[1] += static_cast<unsigned long long>(total_misses);
  }

  // Phase 4: write back the LRU order: stale evictables at the front,
  // freshly loaded misses next, then hits at MRU.
  const int total_evictable = hot_size - total_hits;
  const int remaining_evictable = total_evictable - total_misses;
  for (int i = tid; i < hot_size; i += nthreads) {
    if (i < remaining_evictable) {
      row_lru[i] = s_lru_out[hot_size - 1 - total_misses - i];
    } else if (i < remaining_evictable + total_misses) {
      row_lru[i] = s_lru_out[hot_size - 1 - (i - remaining_evictable)];
    } else {
      row_lru[i] = s_lru_out[i - remaining_evictable - total_misses];
    }
  }
}

#pragma omp end declare target

// Largest of 16, 4, and 1 bytes that divides every address and length a row
// copy touches: both cache bases, both page strides, and the span lengths
// (row_bytes when value_bytes <= 0, otherwise value_bytes and
// row_bytes - value_bytes). Row offsets inside a page are multiples of those
// lengths, so each unit is aligned and never crosses a span.
int copy_unit_bytes(const void* a_base, int64_t a_stride, const void* b_base,
                    int64_t b_stride, int64_t row_bytes, int64_t value_bytes) {
  uintptr_t bits = reinterpret_cast<uintptr_t>(a_base) |
                   reinterpret_cast<uintptr_t>(b_base) |
                   static_cast<uintptr_t>(a_stride) |
                   static_cast<uintptr_t>(b_stride);
  if (value_bytes <= 0) {
    bits |= static_cast<uintptr_t>(row_bytes);
  } else {
    bits |= static_cast<uintptr_t>(value_bytes) |
            static_cast<uintptr_t>(row_bytes - value_bytes);
  }
  if ((bits & 15) == 0) return 16;
  if ((bits & 3) == 0) return 4;
  return 1;
}

// Lanes per copied row in the gather and backup bodies, as a shift. With
// 64-wide AMD wavefronts every load and store instruction then touches one
// contiguous span of one row, which roughly halves host-memory copy time
// against packing several rows into a wavefront.
constexpr int kRowLaneShift = 6;
constexpr int64_t kRowLanes = int64_t{1} << kRowLaneShift;

// Copy one cache row with the lane-th of kRowLanes cooperating iterations:
// value units lane, lane + kRowLanes, ..., then scale units the same way.
// src_row and dst_row are byte offsets of the rows' value spans and
// src_scale and dst_scale those of their scale spans (ignored when
// value_units == units). Unit size T; value_units and units count T units.
template <typename T>
inline void copy_row_lanes(const char* src, int64_t src_row,
                           int64_t src_scale, char* dst, int64_t dst_row,
                           int64_t dst_scale, int64_t value_units,
                           int64_t units, int64_t lane) {
  const T* sv = reinterpret_cast<const T*>(src + src_row);
  T* dv = reinterpret_cast<T*>(dst + dst_row);
  for (int64_t u = lane; u < value_units; u += kRowLanes) dv[u] = sv[u];
  const T* ss = reinterpret_cast<const T*>(src + src_scale);
  T* ds = reinterpret_cast<T*>(dst + dst_scale);
  for (int64_t u = lane; u < units - value_units; u += kRowLanes) {
    ds[u] = ss[u];
  }
}

// Zero one cache row with the lane-th of kRowLanes cooperating iterations,
// with the same span layout as copy_row_lanes.
template <typename T>
inline void zero_row_lanes(char* dst, int64_t dst_row, int64_t dst_scale,
                           int64_t value_units, int64_t units, int64_t lane) {
  T* dv = reinterpret_cast<T*>(dst + dst_row);
  for (int64_t u = lane; u < value_units; u += kRowLanes) dv[u] = T{};
  T* ds = reinterpret_cast<T*>(dst + dst_scale);
  for (int64_t u = lane; u < units - value_units; u += kRowLanes) ds[u] = T{};
}

// gather_plan body with copy unit T. kRowLanes iterations per column. Lane
// 0 of every column writes its attention id: -1 on a padding row (negative
// request_state_indices entry) or a negative hot index, otherwise the
// attention id of the hot index. A column is copied only when the row is not
// padding, its miss flag is nonzero, both indexes are in range, and the host
// index is below host_rows; a host index at or past host_rows zeros the hot
// slot instead.
template <typename T>
void gather_plan_body(const char* host_cache, char* hot_cache,
                      const int32_t* global_indices,
                      const int32_t* hot_indices, const int32_t* miss_mask,
                      int32_t* attention_indices,
                      const int32_t* request_state_indices, int64_t host_rows,
                      int32_t host_block_size, int64_t host_block_stride,
                      int64_t hot_rows, int64_t row_bytes,
                      int64_t row_value_bytes, int64_t hot_block_stride,
                      int32_t hot_block_size, int32_t num_rows, int32_t top_k,
                      int64_t attention_block_stride) {
  const int64_t items = static_cast<int64_t>(num_rows) * top_k;
  const int64_t unit = static_cast<int64_t>(sizeof(T));
  const int64_t units = row_bytes / unit;
  const int64_t value_bytes = row_value_bytes > 0 ? row_value_bytes : row_bytes;
  const int64_t value_units = value_bytes / unit;
  const int64_t threads = items << kRowLaneShift;
  #pragma omp target teams distribute parallel for \
      is_device_ptr(host_cache, hot_cache, global_indices, hot_indices, \
                    miss_mask, attention_indices, request_state_indices)
  for (int64_t t = 0; t < threads; ++t) {
    const int64_t item = t >> kRowLaneShift;
    const int64_t lane = t & (kRowLanes - 1);
    const int64_t row = item / top_k;
    const bool is_padding =
        request_state_indices != nullptr && request_state_indices[row] < 0;
    const int32_t dst = hot_indices[item];
    if (lane == 0 && attention_indices != nullptr) {
      attention_indices[item] =
          is_padding ? -1
                     : (dst < 0 ? -1
                                : (dst / hot_block_size) *
                                          attention_block_stride +
                                      dst % hot_block_size);
    }
    if (is_padding || miss_mask[item] == 0) continue;
    const int32_t g = global_indices[item];
    if (g < 0 || dst < 0 || dst >= hot_rows) continue;
    const int64_t dst_row = row_byte_offset(dst, hot_block_size,
                                            hot_block_stride, row_bytes,
                                            row_value_bytes, 0);
    const int64_t dst_scale = row_byte_offset(dst, hot_block_size,
                                              hot_block_stride, row_bytes,
                                              row_value_bytes, value_bytes);
    if (g < host_rows) {
      copy_row_lanes<T>(
          host_cache,
          row_byte_offset(g, host_block_size, host_block_stride, row_bytes,
                          row_value_bytes, 0),
          row_byte_offset(g, host_block_size, host_block_stride, row_bytes,
                          row_value_bytes, value_bytes),
          hot_cache, dst_row, dst_scale, value_units, units, lane);
    } else {
      // No source row for g: zero the planned slot rather than serving
      // whatever bytes it held.
      zero_row_lanes<T>(hot_cache, dst_row, dst_scale, value_units, units,
                        lane);
    }
  }
}

// gather_compact body with copy unit T. kRowLanes iterations per compact
// entry. Entries at or past the row's miss count, clamped to [0, top_k], are
// ignored. An entry is skipped when the host index is negative or the hot
// index is out of range. A host index below host_rows is copied; one at or
// past host_rows zeros the hot slot.
template <typename T>
void gather_compact_body(const char* host_cache, char* hot_cache,
                         const int32_t* miss_global_indices,
                         const int32_t* miss_hot_indices,
                         const int32_t* miss_counts, int64_t host_rows,
                         int32_t host_block_size, int64_t host_block_stride,
                         int64_t hot_rows, int64_t row_bytes,
                         int64_t row_value_bytes, int64_t hot_block_stride,
                         int32_t hot_block_size, int32_t num_rows,
                         int32_t top_k) {
  const int64_t items = static_cast<int64_t>(num_rows) * top_k;
  const int64_t unit = static_cast<int64_t>(sizeof(T));
  const int64_t units = row_bytes / unit;
  const int64_t value_bytes = row_value_bytes > 0 ? row_value_bytes : row_bytes;
  const int64_t value_units = value_bytes / unit;
  const int64_t threads = items << kRowLaneShift;
  #pragma omp target teams distribute parallel for \
      is_device_ptr(host_cache, hot_cache, miss_global_indices, \
                    miss_hot_indices, miss_counts)
  for (int64_t t = 0; t < threads; ++t) {
    const int64_t item = t >> kRowLaneShift;
    const int64_t lane = t & (kRowLanes - 1);
    const int64_t row = item / top_k;
    const int64_t col = item % top_k;
    int32_t miss_count = miss_counts[row];
    miss_count = miss_count < 0 ? 0 : (miss_count > top_k ? top_k : miss_count);
    if (col >= miss_count) continue;
    const int32_t g = miss_global_indices[item];
    const int32_t dst = miss_hot_indices[item];
    if (g < 0 || dst < 0 || dst >= hot_rows) continue;
    const int64_t dst_row = row_byte_offset(dst, hot_block_size,
                                            hot_block_stride, row_bytes,
                                            row_value_bytes, 0);
    const int64_t dst_scale = row_byte_offset(dst, hot_block_size,
                                              hot_block_stride, row_bytes,
                                              row_value_bytes, value_bytes);
    if (g < host_rows) {
      copy_row_lanes<T>(
          host_cache,
          row_byte_offset(g, host_block_size, host_block_stride, row_bytes,
                          row_value_bytes, 0),
          row_byte_offset(g, host_block_size, host_block_stride, row_bytes,
                          row_value_bytes, value_bytes),
          hot_cache, dst_row, dst_scale, value_units, units, lane);
    } else {
      zero_row_lanes<T>(hot_cache, dst_row, dst_scale, value_units, units,
                        lane);
    }
  }
}

// backup body with copy unit T. kRowLanes iterations per item. Copy
// src_cache[src_indices[i]] into host_cache[dst_slots[i]], skipping the item
// when the source is outside [0, src_rows) or the destination is outside
// [0, host_rows).
template <typename T>
void backup_body(const char* src_cache, const int64_t* src_indices,
                 char* host_cache, const int64_t* dst_slots, int64_t row_bytes,
                 int64_t row_value_bytes, int64_t src_block_stride,
                 int32_t src_block_size, int64_t host_block_stride,
                 int32_t host_block_size, int32_t num_items, int64_t src_rows,
                 int64_t host_rows) {
  const int64_t unit = static_cast<int64_t>(sizeof(T));
  const int64_t units = row_bytes / unit;
  const int64_t value_bytes = row_value_bytes > 0 ? row_value_bytes : row_bytes;
  const int64_t value_units = value_bytes / unit;
  const int64_t threads = static_cast<int64_t>(num_items) << kRowLaneShift;
  #pragma omp target teams distribute parallel for \
      is_device_ptr(src_cache, src_indices, host_cache, dst_slots)
  for (int64_t t = 0; t < threads; ++t) {
    const int64_t i = t >> kRowLaneShift;
    const int64_t lane = t & (kRowLanes - 1);
    const int64_t s = src_indices[i];
    const int64_t d = dst_slots[i];
    if (s < 0 || s >= src_rows || d < 0 || d >= host_rows) continue;
    copy_row_lanes<T>(
        src_cache,
        row_byte_offset(s, src_block_size, src_block_stride, row_bytes,
                        row_value_bytes, 0),
        row_byte_offset(s, src_block_size, src_block_stride, row_bytes,
                        row_value_bytes, value_bytes),
        host_cache,
        row_byte_offset(d, host_block_size, host_block_stride, row_bytes,
                        row_value_bytes, 0),
        row_byte_offset(d, host_block_size, host_block_stride, row_bytes,
                        row_value_bytes, value_bytes),
        value_units, units, lane);
  }
}

}  // namespace

// Scratch sizes for num_rows rows: top_k int32 ids, 2*top_k int32 keys,
// 2*top_k int16 values, hot_size int16 compacted slots, and hot_size int16
// lookup results per row, plus kHisparseResolveThreads int32 prefix-sum
// entries per row (one team never exceeds num_rows teams).
HisparseResolveScratchBytes hisparse_resolve_scratch_bytes(int32_t num_rows,
                                                           int32_t top_k,
                                                           int32_t hot_size) {
  const size_t rows = static_cast<size_t>(num_rows);
  HisparseResolveScratchBytes bytes;
  bytes.topk = rows * static_cast<size_t>(top_k) * sizeof(int32_t);
  bytes.hash_keys = rows * 2 * static_cast<size_t>(top_k) * sizeof(int32_t);
  bytes.hash_vals = rows * 2 * static_cast<size_t>(top_k) * sizeof(int16_t);
  bytes.lru_out = rows * static_cast<size_t>(hot_size) * sizeof(int16_t);
  bytes.found = bytes.lru_out;
  bytes.team = rows * kHisparseResolveThreads * sizeof(int32_t);
  return bytes;
}

// Run resolve_row for every batch row, one row per team of up to
// kHisparseResolveThreads threads; a team handles several rows in turn
// when the runtime grants fewer teams than rows. Return 0 without running
// when num_rows or top_k is 0. Return 1, after printing the reason, when a
// source block table is set without request_ids, hot_size < top_k,
// hot_size > 32768, or a scratch buffer is null.
int hisparse_resolve_residency_launch(const HisparseResolveArgs& args) {
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
  if (args.scratch_topk == nullptr || args.scratch_hash_keys == nullptr ||
      args.scratch_hash_vals == nullptr || args.scratch_lru_out == nullptr ||
      args.scratch_found == nullptr || args.scratch_team == nullptr) {
    fprintf(stderr, "HiSparse resolve scratch buffers are required\n");
    return 1;
  }
  const HisparseResolveArgs a = args;
  const int num_rows = args.num_rows;
  #pragma omp target teams num_teams(num_rows) \
      thread_limit(kHisparseResolveThreads) firstprivate(a)
  #pragma omp parallel num_threads(kHisparseResolveThreads)
  {
    const int team = omp_get_team_num();
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    int32_t* team_scan =
        a.scratch_team + static_cast<int64_t>(team) * kHisparseResolveThreads;
    for (int row = team; row < num_rows; row += omp_get_num_teams()) {
      resolve_row(a, row, tid, nthreads, team_scan);
    }
  }
  return 0;
}

// Run gather_plan with the widest copy unit the layout allows. Return 0
// without running when num_rows or top_k is 0.
int hisparse_gather_plan_launch(
    const char* host_cache, char* hot_cache, const int32_t* global_indices,
    const int32_t* hot_indices, const int32_t* miss_mask,
    int32_t* attention_indices, const int32_t* request_state_indices,
    int64_t host_rows, int32_t host_block_size, int64_t host_block_stride,
    int64_t hot_rows, int64_t row_bytes, int64_t row_value_bytes,
    int64_t hot_block_stride, int32_t hot_block_size, int32_t num_rows,
    int32_t top_k, int64_t attention_block_stride) {
  if (num_rows == 0 || top_k == 0) return 0;
  const int unit = copy_unit_bytes(host_cache, host_block_stride, hot_cache,
                                   hot_block_stride, row_bytes,
                                   row_value_bytes);
  if (unit == 16) {
    gather_plan_body<copy_unit16>(
        host_cache, hot_cache, global_indices, hot_indices, miss_mask,
        attention_indices, request_state_indices, host_rows, host_block_size,
        host_block_stride, hot_rows, row_bytes, row_value_bytes,
        hot_block_stride, hot_block_size, num_rows, top_k,
        attention_block_stride);
  } else if (unit == 4) {
    gather_plan_body<uint32_t>(
        host_cache, hot_cache, global_indices, hot_indices, miss_mask,
        attention_indices, request_state_indices, host_rows, host_block_size,
        host_block_stride, hot_rows, row_bytes, row_value_bytes,
        hot_block_stride, hot_block_size, num_rows, top_k,
        attention_block_stride);
  } else {
    gather_plan_body<char>(
        host_cache, hot_cache, global_indices, hot_indices, miss_mask,
        attention_indices, request_state_indices, host_rows, host_block_size,
        host_block_stride, hot_rows, row_bytes, row_value_bytes,
        hot_block_stride, hot_block_size, num_rows, top_k,
        attention_block_stride);
  }
  return 0;
}

// Run gather_compact with the widest copy unit the layout allows. Return 0
// without running when num_rows or top_k is 0.
int hisparse_gather_compact_launch(
    const char* host_cache, char* hot_cache, const int32_t* miss_global_indices,
    const int32_t* miss_hot_indices, const int32_t* miss_counts,
    int64_t host_rows, int32_t host_block_size, int64_t host_block_stride,
    int64_t hot_rows, int64_t row_bytes, int64_t row_value_bytes,
    int64_t hot_block_stride, int32_t hot_block_size, int32_t num_rows,
    int32_t top_k) {
  if (num_rows == 0 || top_k == 0) return 0;
  const int unit = copy_unit_bytes(host_cache, host_block_stride, hot_cache,
                                   hot_block_stride, row_bytes,
                                   row_value_bytes);
  if (unit == 16) {
    gather_compact_body<copy_unit16>(
        host_cache, hot_cache, miss_global_indices, miss_hot_indices,
        miss_counts, host_rows, host_block_size, host_block_stride, hot_rows,
        row_bytes, row_value_bytes, hot_block_stride, hot_block_size, num_rows,
        top_k);
  } else if (unit == 4) {
    gather_compact_body<uint32_t>(
        host_cache, hot_cache, miss_global_indices, miss_hot_indices,
        miss_counts, host_rows, host_block_size, host_block_stride, hot_rows,
        row_bytes, row_value_bytes, hot_block_stride, hot_block_size, num_rows,
        top_k);
  } else {
    gather_compact_body<char>(
        host_cache, hot_cache, miss_global_indices, miss_hot_indices,
        miss_counts, host_rows, host_block_size, host_block_stride, hot_rows,
        row_bytes, row_value_bytes, hot_block_stride, hot_block_size, num_rows,
        top_k);
  }
  return 0;
}

// Run backup with the widest copy unit the layout allows. Return 0 without
// running when num_items is 0.
int hisparse_backup_launch(
    const char* src_cache, const int64_t* src_indices, char* host_cache,
    const int64_t* dst_slots, int64_t row_bytes, int64_t row_value_bytes,
    int64_t src_block_stride, int32_t src_block_size,
    int64_t host_block_stride, int32_t host_block_size, int32_t num_items,
    int64_t src_rows, int64_t host_rows) {
  if (num_items == 0) return 0;
  const int unit = copy_unit_bytes(src_cache, src_block_stride, host_cache,
                                   host_block_stride, row_bytes,
                                   row_value_bytes);
  if (unit == 16) {
    backup_body<copy_unit16>(src_cache, src_indices, host_cache, dst_slots,
                             row_bytes, row_value_bytes, src_block_stride,
                             src_block_size, host_block_stride,
                             host_block_size, num_items, src_rows, host_rows);
  } else if (unit == 4) {
    backup_body<uint32_t>(src_cache, src_indices, host_cache, dst_slots,
                          row_bytes, row_value_bytes, src_block_stride,
                          src_block_size, host_block_stride, host_block_size,
                          num_items, src_rows, host_rows);
  } else {
    backup_body<char>(src_cache, src_indices, host_cache, dst_slots, row_bytes,
                      row_value_bytes, src_block_stride, src_block_size,
                      host_block_stride, host_block_size, num_items, src_rows,
                      host_rows);
  }
  return 0;
}

// For every token, look up its request and state row once, then have the
// team's threads set each device_global_indices[state][offset] that equals
// the written slot to -1. Nothing is written for a request outside
// [0, num_request_ids), a state outside [0, num_state_rows), or a negative
// written slot. A team handles several tokens in turn when the runtime
// grants fewer teams than tokens. Return 0 without running when num_tokens,
// num_state_rows, or region_stride is 0.
int hisparse_invalidate_written_slots_launch(
    int32_t* device_global_indices, const int32_t* request_state_indices,
    const int32_t* req_id_per_token, const int64_t* written_slots,
    int64_t num_tokens, int64_t num_request_ids, int64_t num_state_rows,
    int64_t region_stride) {
  if (num_tokens == 0 || num_state_rows == 0 || region_stride == 0) {
    return 0;
  }
  constexpr int kThreads = 256;
  constexpr int64_t kMaxTeams = 1 << 16;
  const int num_teams =
      static_cast<int>(num_tokens < kMaxTeams ? num_tokens : kMaxTeams);
  #pragma omp target teams num_teams(num_teams) thread_limit(kThreads) \
      is_device_ptr(device_global_indices, request_state_indices, \
                    req_id_per_token, written_slots)
  #pragma omp parallel num_threads(kThreads)
  {
    const int tid = omp_get_thread_num();
    const int nthreads = omp_get_num_threads();
    for (int64_t token_idx = omp_get_team_num(); token_idx < num_tokens;
         token_idx += omp_get_num_teams()) {
      const int32_t req_idx = req_id_per_token[token_idx];
      if (req_idx < 0 || req_idx >= num_request_ids) continue;
      const int32_t state_idx = request_state_indices[req_idx];
      const int64_t written_slot = written_slots[token_idx];
      if (state_idx < 0 || state_idx >= num_state_rows || written_slot < 0) {
        continue;
      }
      int32_t* row = device_global_indices + state_idx * region_stride;
      for (int64_t offset = tid; offset < region_stride; offset += nthreads) {
        if (row[offset] == written_slot) {
          row[offset] = -1;
        }
      }
    }
  }
  return 0;
}
