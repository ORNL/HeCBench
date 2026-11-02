#include <algorithm>
#include <omp.h>

#pragma omp declare target
int64_t save_blocks_offset(
    int* block_offset,
    int64_t range_start,
    int64_t range_end,
    int64_t block_size,
    int64_t input_block_count,
    int64_t kv_seqlen)
{
  if (range_start >= kv_seqlen) {
    return input_block_count;
  }
  if (range_end > kv_seqlen) {
    range_end = kv_seqlen;
  }

  int64_t current_block_count = input_block_count;
  for (int64_t idx = range_start; idx < range_end; idx += block_size) {
    block_offset[current_block_count++] = static_cast<int>(idx);
  }
  return current_block_count;
}
#pragma omp end declare target

void mergehead (
    const int* q_seqlens,
    const int* kv_seqlens,
    const int* vertical_indexes,   // buffer (max NNZ_V)
    const int* slash_indexes,      // buffer (max NNZ_S)
    const int* per_head_vertical_topkv,
    const int* per_head_slash_topkv,
    int* block_count,
    int* block_offset,
    int* column_count,
    int* column_index,
    int64_t BATCH,
    int64_t N_HEADS,
    int64_t N_ROWS,
    int64_t BLOCK_SIZE_M,
    int64_t BLOCK_SIZE_N,
    int64_t MAX_NNZ_V,
    int64_t MAX_NNZ_S,
    bool causal)
{
  #pragma omp target teams distribute parallel for collapse(3) //\
   num_teams(BATCH * N_HEADS * N_ROWS) num_threads(64)
  for (int64_t b = 0; b < BATCH; ++b) {
    for (int64_t h = 0; h < N_HEADS; ++h) {
      for (int64_t block_idx_m = 0; block_idx_m < N_ROWS; ++block_idx_m) {
        int64_t q_seqlen  = q_seqlens[b];
        int64_t kv_seqlen = kv_seqlens[b];

        int64_t NNZ_V = per_head_vertical_topkv[h];
        int64_t NNZ_S = per_head_slash_topkv[h];

        const int* v_base = vertical_indexes + (b * N_HEADS + h) * MAX_NNZ_V;
        const int* s_base = slash_indexes + (b * N_HEADS + h) * MAX_NNZ_S;


        int64_t start_m = block_idx_m * BLOCK_SIZE_M;
        if (start_m >= q_seqlen) continue;

        int64_t end_m = start_m + BLOCK_SIZE_M;

        int64_t row_offset = (b * N_HEADS + h) * N_ROWS + block_idx_m;

        int* block_count_ptr  = block_count  + row_offset;
        int* column_count_ptr = column_count + row_offset;
        int* block_offset_ptr = block_offset + row_offset * MAX_NNZ_S;
        int* column_index_ptr = column_index + row_offset * MAX_NNZ_V;

        bool has_slash = true;
        int64_t tmp_col_cnt = 0, tmp_blk_cnt = 0;
        int64_t s = 0, v = 0;

        int64_t v_idx = v_base[v++];
        int64_t s_idx = s_base[s++];

        if (causal) {
          while (s_idx >= end_m + (kv_seqlen - q_seqlen) && s < NNZ_S)
            s_idx = s_base[s++];

          if (s_idx > end_m + (kv_seqlen - q_seqlen))
            has_slash = false;

          s_idx = std::max(
              (kv_seqlen - q_seqlen) + end_m - s_idx,
              BLOCK_SIZE_M);
        } else {

          if (s_idx > end_m + kv_seqlen)
            has_slash = false;

          s_idx = std::max(
              kv_seqlen + end_m - s_idx,
              BLOCK_SIZE_M);
        }

        int64_t range_start = s_idx - BLOCK_SIZE_M;
        int64_t range_end   = s_idx;

        if (!has_slash) {
          if (causal) {
            range_start = (kv_seqlen - q_seqlen) + end_m;
            range_end   = range_start + BLOCK_SIZE_N;
          } else {
            range_start = kv_seqlen;
            range_end   = kv_seqlen + BLOCK_SIZE_N;
          }
        }

        bool slash_finished = false;

        while (true) {
          if (v_idx < range_end) {
            if (v_idx < range_start) {
              column_index_ptr[tmp_col_cnt++] = (int)v_idx;
            }

            if (v < NNZ_V) {
              v_idx = v_base[v++];
            } else {
              v_idx = causal
                ? end_m + BLOCK_SIZE_N + (kv_seqlen - q_seqlen)
                : end_m + BLOCK_SIZE_N + kv_seqlen;
            }

          } else {
            if ((s < NNZ_S && causal) ||
                (s < NNZ_S && !causal && s_base[s] >= start_m)) {

              if (causal)
                s_idx = std::max(
                    (kv_seqlen - q_seqlen) + end_m - s_base[s++],
                    BLOCK_SIZE_M);
              else
                s_idx = std::max(
                    kv_seqlen + end_m - s_base[s++],
                    BLOCK_SIZE_M);

            } else {
              if (v == NNZ_V || (v_idx > range_start && causal)) {

                if (v == NNZ_V && !causal && v_idx < kv_seqlen)
                  column_index_ptr[tmp_col_cnt++] = (int)v_idx;

                tmp_blk_cnt = save_blocks_offset(
                    block_offset_ptr,
                    range_start,
                    range_end,
                    BLOCK_SIZE_N,
                    tmp_blk_cnt,
                    kv_seqlen);

                break;
              } else {
                if (causal) {
                  range_start = (kv_seqlen - q_seqlen) + end_m;
                  range_end   = range_start + BLOCK_SIZE_N;
                } else {
                  tmp_blk_cnt = save_blocks_offset(
                      block_offset_ptr,
                      range_start,
                      range_end,
                      BLOCK_SIZE_N,
                      tmp_blk_cnt,
                      kv_seqlen);

                  range_start = kv_seqlen;
                  range_end   = kv_seqlen + BLOCK_SIZE_N;
                }
                slash_finished = true;
              }
            }

            if (!slash_finished) {
              if (s_idx > range_end + BLOCK_SIZE_M) {
                tmp_blk_cnt = save_blocks_offset(
                    block_offset_ptr,
                    range_start,
                    range_end,
                    BLOCK_SIZE_N,
                    tmp_blk_cnt,
                    kv_seqlen);

                range_start = s_idx - BLOCK_SIZE_M;
                range_end   = s_idx;

              } else if (s_idx > range_end) {
                range_end += BLOCK_SIZE_M;
              }
            }
          }
        }

        block_count_ptr[0]  = (int)tmp_blk_cnt;
        column_count_ptr[0] = (int)tmp_col_cnt;
      }
    }
  }
}
