// CPU reference for one HiSparse decode step.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <vector>

// Hot-cache row for slot: block id hot_block_table[row * table_stride +
// slot / block_size], then that id times block_size plus slot % block_size.
// The sum is computed in int64 and returned as int32. No bounds check.
inline int32_t physical_hot_row(const int32_t* hot_block_table, int32_t row,
                                int64_t table_stride, int32_t block_size,
                                int32_t slot) {
  const int32_t block =
      hot_block_table[static_cast<int64_t>(row) * table_stride +
                      slot / block_size];
  return static_cast<int32_t>(static_cast<int64_t>(block) * block_size +
                              slot % block_size);
}

// Attention id for a hot row. Returns -1 when physical_row < 0. Otherwise
// returns (physical_row / hot_block_size) * attention_block_stride +
// physical_row % hot_block_size, computed in int64 and cast to int32.
inline int32_t attention_of(int32_t physical_row, int32_t hot_block_size,
                            int64_t attention_block_stride) {
  if (physical_row < 0) return -1;
  const int64_t value =
      static_cast<int64_t>(physical_row / hot_block_size) *
          attention_block_stride +
      physical_row % hot_block_size;
  return static_cast<int32_t>(value);
}

// Bytes covering rows cache rows. Returns 0 when rows <= 0. Otherwise
// ceil(rows / block_size) pages times block_stride.
inline int64_t cache_bytes(int64_t rows, int32_t block_size,
                           int64_t block_stride) {
  if (rows <= 0) return 0;
  const int64_t pages = (rows + block_size - 1) / block_size;
  return pages * block_stride;
}

// Byte address of row: page (row / block_size) at block_stride, plus
// (row % block_size) * row_bytes. No bounds check.
inline char* cache_row_ptr(char* cache, int64_t row, int32_t block_size,
                           int64_t block_stride, int64_t row_bytes) {
  return cache + (row / block_size) * block_stride +
         (row % block_size) * row_bytes;
}

// Const overload of cache_row_ptr. Same address, no bounds check.
inline const char* cache_row_ptr(const char* cache, int64_t row,
                                 int32_t block_size, int64_t block_stride,
                                 int64_t row_bytes) {
  return cache + (row / block_size) * block_stride +
         (row % block_size) * row_bytes;
}

// Host copy of one cache row, matching copy_cache_row_warp. value_bytes <= 0
// copies row_bytes as one span. Otherwise the value span and the scale span
// (row_bytes - value_bytes) are copied separately.
inline void copy_cache_row(const char* src_cache, int64_t src_row,
                           int32_t src_block_size, int64_t src_block_stride,
                           char* dst_cache, int64_t dst_row,
                           int32_t dst_block_size, int64_t dst_block_stride,
                           int64_t row_bytes, int64_t value_bytes) {
  if (value_bytes <= 0) {
    std::memcpy(cache_row_ptr(dst_cache, dst_row, dst_block_size,
                              dst_block_stride, row_bytes),
                cache_row_ptr(src_cache, src_row, src_block_size,
                              src_block_stride, row_bytes),
                static_cast<size_t>(row_bytes));
    return;
  }
  const int64_t scale_bytes = row_bytes - value_bytes;
  const int64_t src_offset = src_row % src_block_size;
  const int64_t dst_offset = dst_row % dst_block_size;
  const char* src_page =
      src_cache + (src_row / src_block_size) * src_block_stride;
  char* dst_page = dst_cache + (dst_row / dst_block_size) * dst_block_stride;
  std::memcpy(dst_page + dst_offset * value_bytes,
              src_page + src_offset * value_bytes,
              static_cast<size_t>(value_bytes));
  std::memcpy(dst_page + dst_block_size * value_bytes + dst_offset * scale_bytes,
              src_page + src_block_size * value_bytes + src_offset * scale_bytes,
              static_cast<size_t>(scale_bytes));
}

// Host zero of one cache row, matching zero_cache_row_warp. value_bytes <= 0
// zeros row_bytes as one span. Otherwise the value and scale spans are
// zeroed separately.
inline void zero_cache_row(char* cache, int64_t row, int32_t block_size,
                           int64_t block_stride, int64_t row_bytes,
                           int64_t value_bytes) {
  if (value_bytes <= 0) {
    std::memset(cache_row_ptr(cache, row, block_size, block_stride, row_bytes),
                0, static_cast<size_t>(row_bytes));
    return;
  }
  const int64_t scale_bytes = row_bytes - value_bytes;
  const int64_t offset = row % block_size;
  char* page = cache + (row / block_size) * block_stride;
  std::memset(page + offset * value_bytes, 0, static_cast<size_t>(value_bytes));
  std::memset(page + block_size * value_bytes + offset * scale_bytes, 0,
              static_cast<size_t>(scale_bytes));
}

// CPU mirror of hisparse_resolve_residency_kernel for one batch. Writes the
// hot indexes, optional attention indexes, miss flags, resolved ids, optional
// valid counts, compact swap lists, device-global ids, LRU list, and
// stats[0] hits / stats[1] misses. request_ids, request_state_indices,
// attention_indices, miss_mask, and valid_counts may be null. A source block
// table reads request_ids[row] with no null check. Rows that finish before
// the LRU walk do not add to stats. Returns false, after printing, when a
// row's hit, miss, and evictable counts disagree.
inline bool hisparse_resolve_reference(
    const int32_t* hot_block_table, const int32_t* global_indices,
    const int32_t* request_ids, const int32_t* source_block_table,
    const int32_t* resident_block_table, int32_t* hot_indices,
    int32_t* attention_indices, int32_t* miss_mask, int32_t* resolved,
    int32_t* valid_counts, int32_t* swap_host, int32_t* swap_device,
    int32_t* swap_counts, int32_t* dgi, int16_t* lru,
    const int32_t* request_state_indices, int32_t request_state_count,
    unsigned long long* stats, int32_t num_rows, int32_t top_k,
    int32_t hot_size, int64_t host_rows, int64_t hot_table_stride,
    int32_t hot_block_size, int64_t region_stride,
    int64_t attention_block_stride, int64_t source_bt_stride,
    int32_t source_num_reqs, int32_t source_num_blocks,
    int32_t source_block_size, int64_t resident_bt_stride,
    int32_t resident_num_reqs, int32_t resident_num_blocks,
    int32_t resident_block_size, int32_t resident_null_block,
    int64_t input_row_stride, int64_t attention_row_stride,
    int64_t valid_count_stride) {
  stats[0] = 0;
  stats[1] = 0;
  std::unordered_map<int32_t, int> want;
  want.reserve(static_cast<size_t>(top_k) * 2);
  std::vector<int16_t> hit_slots(static_cast<size_t>(hot_size));
  std::vector<int16_t> evict_slots(static_cast<size_t>(hot_size));
  std::vector<char> done(static_cast<size_t>(top_k));

  for (int32_t row = 0; row < num_rows; ++row) {
    int32_t* row_swap_host = swap_host + static_cast<int64_t>(row) * top_k;
    int32_t* row_swap_device = swap_device + static_cast<int64_t>(row) * top_k;
    for (int32_t col = 0; col < top_k; ++col) {
      row_swap_host[col] = -2;
      row_swap_device[col] = -2;
    }

    const int32_t request_row =
        request_ids != nullptr ? request_ids[row] : row;
    int32_t state_row;
    if (request_state_indices != nullptr && request_row >= 0 &&
        request_row < request_state_count) {
      state_row = request_state_indices[request_row];
    } else if (request_state_indices == nullptr) {
      state_row = request_row;
    } else {
      state_row = -1;
    }
    if (state_row < 0) {
      for (int32_t i = 0; i < top_k; ++i) {
        hot_indices[static_cast<int64_t>(row) * top_k + i] = -1;
        if (attention_indices != nullptr) {
          attention_indices[static_cast<int64_t>(row) * attention_row_stride +
                            i] = -1;
        }
        if (resolved != nullptr) {
          resolved[static_cast<int64_t>(row) * top_k + i] = -1;
        }
      }
      if (valid_counts != nullptr) {
        valid_counts[static_cast<int64_t>(row) * valid_count_stride] = 0;
      }
      swap_counts[row] = 0;
      continue;
    }

    const int32_t* row_topk =
        global_indices + static_cast<int64_t>(row) * input_row_stride;
    int32_t* row_out = hot_indices + static_cast<int64_t>(row) * top_k;
    int32_t* row_attn =
        attention_indices != nullptr
            ? attention_indices +
                  static_cast<int64_t>(row) * attention_row_stride
            : nullptr;
    int32_t* row_miss = miss_mask != nullptr
                            ? miss_mask + static_cast<int64_t>(row) * top_k
                            : nullptr;
    int32_t* row_resolved = resolved + static_cast<int64_t>(row) * top_k;
    int32_t* row_dgi = dgi + static_cast<int64_t>(state_row) * region_stride;
    int16_t* row_lru = lru + static_cast<int64_t>(state_row) * hot_size;

    int phase1 = 0;
    int valid = 0;
    for (int32_t i = 0; i < top_k; ++i) {
      const int32_t token_index = row_topk[i];
      int32_t g = token_index;
      int32_t resident_row_id = -1;
      if (source_block_table != nullptr) {
        const int32_t request_id = request_ids[row];
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
              resident_row_id = physical_block * resident_block_size +
                                token_index % resident_block_size;
            }
          }
        }
      }
      if (static_cast<int64_t>(g) >= host_rows) g = -1;
      row_resolved[i] = g;
      if (row_miss != nullptr) row_miss[i] = 0;
      if (resident_row_id >= 0 || g >= 0) valid++;
      if (resident_row_id >= 0) {
        row_out[i] = resident_row_id;
        if (row_attn != nullptr) {
          row_attn[i] = attention_of(resident_row_id, hot_block_size,
                                     attention_block_stride);
        }
        done[static_cast<size_t>(i)] = 1;
        phase1++;
      } else if (g < 0) {
        row_out[i] = -1;
        if (row_attn != nullptr) row_attn[i] = -1;
        done[static_cast<size_t>(i)] = 1;
        phase1++;
      } else {
        done[static_cast<size_t>(i)] = 0;
      }
    }
    if (valid_counts != nullptr) {
      valid_counts[static_cast<int64_t>(row) * valid_count_stride] = valid;
    }
    if (phase1 == top_k) {
      swap_counts[row] = 0;
      continue;
    }

    want.clear();
    for (int32_t i = 0; i < top_k; ++i) {
      if (!done[static_cast<size_t>(i)] && row_resolved[i] >= 0) {
        want[row_resolved[i]] = i;
      }
    }

    int n_hit = 0;
    int n_evict = 0;
    for (int32_t pos = 0; pos < hot_size; ++pos) {
      const int16_t slot = row_lru[pos];
      const int32_t cached =
          (slot >= 0 && slot < region_stride) ? row_dgi[slot] : -1;
      auto it = cached >= 0 ? want.find(cached) : want.end();
      if (it != want.end()) {
        const int idx = it->second;
        done[static_cast<size_t>(idx)] = 1;
        hit_slots[static_cast<size_t>(n_hit++)] = slot;
        const int32_t physical = physical_hot_row(
            hot_block_table, request_row, hot_table_stride, hot_block_size,
            slot);
        row_out[idx] = physical;
        if (row_attn != nullptr) {
          row_attn[idx] =
              attention_of(physical, hot_block_size, attention_block_stride);
        }
      } else {
        evict_slots[static_cast<size_t>(n_evict++)] = slot;
      }
    }

    const int total_hits = n_hit;
    const int total_misses = top_k - total_hits - phase1;
    const int total_evictable = hot_size - total_hits;
    const int remaining = total_evictable - total_misses;
    if (total_misses < 0 || remaining < 0 || n_evict != total_evictable) {
      fprintf(stderr,
              "reference invariant failed on row %d (hits=%d misses=%d "
              "evict=%d phase1=%d)\n",
              row, total_hits, total_misses, n_evict, phase1);
      return false;
    }

    int m = 0;
    for (int32_t i = 0; i < top_k; ++i) {
      if (done[static_cast<size_t>(i)]) continue;
      const int16_t evict_slot = evict_slots[static_cast<size_t>(m)];
      const int32_t g = row_resolved[i];
      if (evict_slot < 0 || evict_slot >= region_stride) {
        row_out[i] = -1;
        if (row_attn != nullptr) row_attn[i] = -1;
        row_swap_host[m] = g;
        row_swap_device[m] = -1;
      } else {
        const int32_t physical = physical_hot_row(
            hot_block_table, request_row, hot_table_stride, hot_block_size,
            evict_slot);
        row_out[i] = physical;
        if (row_attn != nullptr) {
          row_attn[i] =
              attention_of(physical, hot_block_size, attention_block_stride);
        }
        if (row_miss != nullptr) row_miss[i] = 1;
        row_dgi[evict_slot] = g;
        row_swap_host[m] = g;
        row_swap_device[m] = physical;
      }
      m++;
    }
    if (m != total_misses) {
      fprintf(stderr, "miss count mismatch on row %d: walked %d formula %d\n",
              row, m, total_misses);
      return false;
    }
    swap_counts[row] = total_misses;
    stats[0] += static_cast<unsigned long long>(total_hits);
    stats[1] += static_cast<unsigned long long>(total_misses);

    for (int32_t i = 0; i < hot_size; ++i) {
      if (i < remaining) {
        row_lru[i] = evict_slots[static_cast<size_t>(total_misses + i)];
      } else if (i < remaining + total_misses) {
        row_lru[i] = evict_slots[static_cast<size_t>(i - remaining)];
      } else {
        row_lru[i] =
            hit_slots[static_cast<size_t>(i - remaining - total_misses)];
      }
    }
  }
  return true;
}

// CPU mirror of hisparse_backup_kernel. For each item, skip a source outside
// [0, src_rows) or a destination outside [0, host_rows); otherwise copy that
// source row onto the host row.
inline void hisparse_backup_reference(
    const char* src_cache, char* host_cache, const int64_t* src_indices,
    const int64_t* dst_slots, int32_t num_items, int64_t src_rows,
    int64_t host_rows, int32_t src_block_size, int64_t src_block_stride,
    int32_t host_block_size, int64_t host_block_stride, int64_t row_bytes,
    int64_t value_bytes) {
  for (int32_t i = 0; i < num_items; ++i) {
    const int64_t src = src_indices[i];
    const int64_t dst = dst_slots[i];
    if (src < 0 || src >= src_rows || dst < 0 || dst >= host_rows) continue;
    copy_cache_row(src_cache, src, src_block_size, src_block_stride, host_cache,
                   dst, host_block_size, host_block_stride, row_bytes,
                   value_bytes);
  }
}

// CPU mirror of hisparse_invalidate_written_slots_kernel. For each token,
// skip an out-of-range request, an out-of-range state, or a negative written
// slot. Otherwise set every matching device-global id in that state row to -1.
inline void hisparse_invalidate_reference(
    int32_t* dgi, const int32_t* request_state_indices,
    const int32_t* req_id_per_token, const int64_t* written_slots,
    int64_t num_tokens, int64_t num_request_ids, int64_t num_state_rows,
    int64_t region_stride) {
  for (int64_t token = 0; token < num_tokens; ++token) {
    const int32_t req = req_id_per_token[token];
    if (req < 0 || static_cast<int64_t>(req) >= num_request_ids) continue;
    const int32_t state = request_state_indices[req];
    const int64_t written = written_slots[token];
    if (state < 0 || static_cast<int64_t>(state) >= num_state_rows ||
        written < 0) {
      continue;
    }
    int32_t* row = dgi + static_cast<int64_t>(state) * region_stride;
    for (int64_t offset = 0; offset < region_stride; ++offset) {
      if (static_cast<int64_t>(row[offset]) == written) row[offset] = -1;
    }
  }
}

// CPU mirror of hisparse_gather_plan_kernel. batch_state may be null; a
// negative entry makes that row padding. attention_indices may be null.
// miss_mask is required. A column is copied only when the row is not
// padding, the miss flag is nonzero, and both indexes are in range. A host
// index at or past host_rows zeros the hot slot.
inline void hisparse_gather_plan_reference(
    const char* host_cache, char* hot_cache, const int32_t* global_indices,
    const int32_t* hot_indices, const int32_t* miss_mask,
    int32_t* attention_indices, const int32_t* batch_state, int32_t num_rows,
    int32_t top_k, int64_t host_rows, int32_t host_block_size,
    int64_t host_block_stride, int64_t hot_rows, int64_t row_bytes,
    int64_t value_bytes, int64_t hot_block_stride, int32_t hot_block_size,
    int64_t attention_block_stride) {
  for (int32_t row = 0; row < num_rows; ++row) {
    const bool is_padding =
        batch_state != nullptr && batch_state[row] < 0;
    for (int32_t col = 0; col < top_k; ++col) {
      const int64_t index = static_cast<int64_t>(row) * top_k + col;
      const int32_t dst = hot_indices[index];
      if (attention_indices != nullptr) {
        attention_indices[index] =
            is_padding ? -1
                       : attention_of(dst, hot_block_size,
                                      attention_block_stride);
      }
      if (is_padding || miss_mask[index] == 0) continue;
      const int32_t g = global_indices[index];
      if (g < 0 || dst < 0 || static_cast<int64_t>(dst) >= hot_rows) continue;
      if (static_cast<int64_t>(g) < host_rows) {
        copy_cache_row(host_cache, g, host_block_size, host_block_stride,
                       hot_cache, dst, hot_block_size, hot_block_stride,
                       row_bytes, value_bytes);
      } else {
        zero_cache_row(hot_cache, dst, hot_block_size, hot_block_stride,
                       row_bytes, value_bytes);
      }
    }
  }
}

// CPU mirror of hisparse_gather_compact_kernel. Each row's count is clamped
// to [0, top_k]. Those leading compact entries are skipped when the host
// index is negative or the hot index is out of range, copied when the host
// index is below host_rows, and zeroed otherwise.
inline void hisparse_gather_compact_reference(
    const char* host_cache, char* hot_cache, const int32_t* swap_host,
    const int32_t* swap_device, const int32_t* swap_counts, int32_t num_rows,
    int32_t top_k, int64_t host_rows, int32_t host_block_size,
    int64_t host_block_stride, int64_t hot_rows, int64_t row_bytes,
    int64_t value_bytes, int64_t hot_block_stride, int32_t hot_block_size) {
  for (int32_t row = 0; row < num_rows; ++row) {
    int miss_count = swap_counts[row];
    if (miss_count < 0) miss_count = 0;
    if (miss_count > top_k) miss_count = top_k;
    for (int col = 0; col < miss_count; ++col) {
      const int64_t index = static_cast<int64_t>(row) * top_k + col;
      const int32_t g = swap_host[index];
      const int32_t dst = swap_device[index];
      if (g < 0 || dst < 0 || static_cast<int64_t>(dst) >= hot_rows) continue;
      if (static_cast<int64_t>(g) < host_rows) {
        copy_cache_row(host_cache, g, host_block_size, host_block_stride,
                       hot_cache, dst, hot_block_size, hot_block_stride,
                       row_bytes, value_bytes);
      } else {
        zero_cache_row(hot_cache, dst, hot_block_size, hot_block_stride,
                       row_bytes, value_bytes);
      }
    }
  }
}
