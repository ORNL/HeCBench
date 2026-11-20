// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the SGLang project
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <cuda_runtime.h>
#include <cstdint>

// Raw-pointer launch arguments for hisparse_resolve_residency_kernel.
// Nullable index tables (request_ids, source/resident block tables,
// request_state_indices, stats, and the optional outputs) may be nullptr.
// source_block_table requires request_ids.
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
};

// Dynamic shared memory used by hisparse_resolve_residency_kernel when
// hot_size >= top_k. See hisparse_resolve_smem_bytes for the layout.
// Does not validate the arguments.
size_t hisparse_resolve_smem_bytes(int32_t top_k, int32_t hot_size);

// Launch resolve on stream. Returns cudaSuccess without launching when
// num_rows or top_k is 0. Returns cudaErrorInvalidValue when request_ids is
// missing for a source block table, hot_size < top_k, hot_size > 32768, or
// the dynamic shared memory does not fit in an int.
cudaError_t hisparse_resolve_residency_launch(const HisparseResolveArgs& args,
                                              cudaStream_t stream);

// Launch gather_plan on stream. Returns cudaSuccess without launching when
// num_rows or top_k is 0.
cudaError_t hisparse_gather_plan_launch(
    const char* host_cache, char* hot_cache, const int32_t* global_indices,
    const int32_t* hot_indices, const int32_t* miss_mask,
    int32_t* attention_indices, const int32_t* request_state_indices,
    int64_t host_rows, int32_t host_block_size, int64_t host_block_stride,
    int64_t hot_rows, int64_t row_bytes, int64_t row_value_bytes,
    int64_t hot_block_stride, int32_t hot_block_size, int32_t num_rows,
    int32_t top_k, int64_t attention_block_stride, cudaStream_t stream);

// Launch gather_compact on stream. Returns cudaSuccess without launching
// when num_rows or top_k is 0.
cudaError_t hisparse_gather_compact_launch(
    const char* host_cache, char* hot_cache, const int32_t* miss_global_indices,
    const int32_t* miss_hot_indices, const int32_t* miss_counts,
    int64_t host_rows, int32_t host_block_size, int64_t host_block_stride,
    int64_t hot_rows, int64_t row_bytes, int64_t row_value_bytes,
    int64_t hot_block_stride, int32_t hot_block_size, int32_t num_rows,
    int32_t top_k, cudaStream_t stream);

// Launch backup on stream. Returns cudaSuccess without launching when
// num_items is 0.
cudaError_t hisparse_backup_launch(const char* src_cache, const int64_t* src_indices,
                            char* host_cache, const int64_t* dst_slots,
                            int64_t row_bytes, int64_t row_value_bytes,
                            int64_t src_block_stride, int32_t src_block_size,
                            int64_t host_block_stride, int32_t host_block_size,
                            int32_t num_items, int64_t src_rows,
                            int64_t host_rows, cudaStream_t stream);

// Launch invalidation on stream, one block per token. Returns cudaSuccess
// without launching when num_tokens, num_state_rows, or region_stride is 0.
cudaError_t hisparse_invalidate_written_slots_launch(
    int32_t* device_global_indices, const int32_t* request_state_indices,
    const int32_t* req_id_per_token, const int64_t* written_slots,
    int64_t num_tokens, int64_t num_request_ids, int64_t num_state_rows,
    int64_t region_stride, cudaStream_t stream);
