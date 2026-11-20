// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the SGLang project
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <cstddef>
#include <cstdint>

// Raw-pointer launch arguments for the resolve kernel. Every pointer is a
// device address on the default OpenMP device (omp_target_alloc, or host
// memory the device can access). Nullable index tables (request_ids,
// source/resident block tables, request_state_indices, stats, and the
// optional outputs) may be nullptr. source_block_table requires request_ids.
// The scratch_* buffers are per-row work space sized by
// hisparse_resolve_scratch_bytes and are required.
struct HisparseResolveArgs {
  const int32_t* hot_block_table;
  const int32_t* global_indices;
  const int32_t* request_ids;
  const int32_t* source_block_table;
  const int32_t* resident_block_table;
  int32_t* resolved_global_indices;
  int32_t* valid_counts;
  int32_t* swap_host_physical_rows;
  int32_t* swap_device_physical_rows;
  int32_t* swap_counts;
  int32_t* hot_indices;
  int32_t* attention_indices;
  int32_t* miss_mask;
  int32_t* device_global_indices;
  int16_t* lru_slots;
  unsigned long long* stats;
  const int32_t* request_state_indices;
  int32_t request_state_count;
  int64_t host_rows;
  int64_t hot_table_stride;
  int32_t hot_block_size;
  int32_t top_k;
  int32_t hot_size;
  int64_t region_stride;
  int64_t attention_block_stride;
  int64_t source_bt_stride;
  int32_t source_num_reqs;
  int32_t source_num_blocks;
  int32_t source_block_size;
  int64_t resident_bt_stride;
  int32_t resident_num_reqs;
  int32_t resident_num_blocks;
  int32_t resident_block_size;
  int32_t resident_null_block;
  int64_t input_row_stride;
  int64_t attention_row_stride;
  int64_t valid_count_stride;
  int32_t num_rows;
  // [num_rows, top_k] pending ids.
  int32_t* scratch_topk;
  // [num_rows, 2 * top_k] open-addressing keys and top-k positions.
  int32_t* scratch_hash_keys;
  int16_t* scratch_hash_vals;
  // [num_rows, hot_size] compacted slots: hits forward, evictables backward.
  int16_t* scratch_lru_out;
  // [num_rows, hot_size] top-k position owning each LRU slot, or -1.
  int16_t* scratch_found;
  // [num_rows, kHisparseResolveThreads] per-team prefix-sum work space.
  int32_t* scratch_team;
};

// Threads per resolve team.
constexpr int kHisparseResolveThreads = 256;

// Bytes of each resolve scratch buffer for num_rows rows: top-k ids,
// hash keys, hash values, compacted LRU slots, LRU lookup results, and
// prefix-sum work space, in that order.
struct HisparseResolveScratchBytes {
  size_t topk;
  size_t hash_keys;
  size_t hash_vals;
  size_t lru_out;
  size_t found;
  size_t team;
};
HisparseResolveScratchBytes hisparse_resolve_scratch_bytes(int32_t num_rows,
                                                           int32_t top_k,
                                                           int32_t hot_size);

// Every launcher runs a synchronous target region on the default device and
// returns 0 when it ran, or did nothing because there was no work. A nonzero
// return means the arguments were rejected and nothing ran.

// Launch resolve. Returns 1 when request_ids is missing for a source block
// table, hot_size < top_k, hot_size > 32768, or a scratch buffer is null.
int hisparse_resolve_residency_launch(const HisparseResolveArgs& args);

// Launch gather_plan. Does nothing when num_rows or top_k is 0.
int hisparse_gather_plan_launch(
    const char* host_cache, char* hot_cache, const int32_t* global_indices,
    const int32_t* hot_indices, const int32_t* miss_mask,
    int32_t* attention_indices, const int32_t* request_state_indices,
    int64_t host_rows, int32_t host_block_size, int64_t host_block_stride,
    int64_t hot_rows, int64_t row_bytes, int64_t row_value_bytes,
    int64_t hot_block_stride, int32_t hot_block_size, int32_t num_rows,
    int32_t top_k, int64_t attention_block_stride);

// Launch gather_compact. Does nothing when num_rows or top_k is 0.
int hisparse_gather_compact_launch(
    const char* host_cache, char* hot_cache, const int32_t* miss_global_indices,
    const int32_t* miss_hot_indices, const int32_t* miss_counts,
    int64_t host_rows, int32_t host_block_size, int64_t host_block_stride,
    int64_t hot_rows, int64_t row_bytes, int64_t row_value_bytes,
    int64_t hot_block_stride, int32_t hot_block_size, int32_t num_rows,
    int32_t top_k);

// Launch backup. Does nothing when num_items is 0.
int hisparse_backup_launch(
    const char* src_cache, const int64_t* src_indices, char* host_cache,
    const int64_t* dst_slots, int64_t row_bytes, int64_t row_value_bytes,
    int64_t src_block_stride, int32_t src_block_size,
    int64_t host_block_stride, int32_t host_block_size, int32_t num_items,
    int64_t src_rows, int64_t host_rows);

// Launch invalidation. Does nothing when num_tokens, num_state_rows, or
// region_stride is 0.
int hisparse_invalidate_written_slots_launch(
    int32_t* device_global_indices, const int32_t* request_state_indices,
    const int32_t* req_id_per_token, const int64_t* written_slots,
    int64_t num_tokens, int64_t num_request_ids, int64_t num_state_rows,
    int64_t region_stride);
