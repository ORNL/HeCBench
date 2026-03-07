/*
 * Adapted from
 * https://github.com/NVIDIA/FasterTransformer/blob/release/v5.3_tag/src/fastertransformer/kernels/decoder_masked_multihead_attention/decoder_masked_multihead_attention_template.hpp
 * https://github.com/vllm-project/vllm/blob/main/benchmarks/kernels/benchmark_paged_attention.py
 *
 *
 * Copyright (c) 2023, The vLLM team.
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <chrono>
#include "attention_kernels.cuh"
#include "cuda_compat.h"
#include "kvcache.h"
#include "reference.h"
#include <cstdlib>


#define LAUNCH_PAGED_ATTENTION_V1(HEAD_SIZE)                          \
  VLLM_DevFuncAttribute_SET_MaxDynamicSharedMemorySize(               \
      ((void*)paged_attention_v1_kernel<T, CACHE_T, HEAD_SIZE,        \
                                              BLOCK_SIZE, NUM_THREADS,\
                                              IS_BLOCK_SPARSE>),      \
      shared_mem_size);                                               \
  paged_attention_v1_kernel<T, CACHE_T, HEAD_SIZE, BLOCK_SIZE,        \
                                  NUM_THREADS, IS_BLOCK_SPARSE>       \
      <<<grid, block, shared_mem_size, 0>>>(                          \
          out, query, key_cache, value_cache, num_kv_heads,           \
          scale, block_tables, seq_lens, max_num_blocks_per_seq,      \
          alibi_slopes, q_stride, kv_block_stride, kv_head_stride,    \
          tp_rank, blocksparse_local_blocks,                          \
          blocksparse_vert_stride, blocksparse_block_size,            \
          blocksparse_head_sliding_step);

// TODO(woosuk): Tune NUM_THREADS.
template <typename T, typename CACHE_T, int BLOCK_SIZE,
          bool IS_BLOCK_SPARSE, int NUM_THREADS = 128>
void paged_attention_v1_launcher(
    T *out,
    T *query,
    CACHE_T* key_cache,
    CACHE_T* value_cache,
    int num_kv_heads,
    float scale,
    int *block_tables,
    int *seq_lens,
    int max_seq_len,
    float *alibi_slopes,
    const int tp_rank,
    const int blocksparse_local_blocks, const int blocksparse_vert_stride,
    const int blocksparse_block_size, const int blocksparse_head_sliding_step,
    const int num_seqs,
    const int num_heads,
    const int head_size,
    const int max_num_blocks_per_seq,
    const int q_stride,
    const int kv_block_stride,
    const int kv_head_stride)
{
  const int NUM_WARPS = NUM_THREADS / WARP_SIZE;
  int padded_max_seq_len = (max_seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE;
  int logits_size = padded_max_seq_len * sizeof(float);
  int outputs_size = (NUM_WARPS / 2) * head_size * sizeof(float);
  int shared_mem_size = std::max(logits_size, outputs_size);

  dim3 grid(num_heads, num_seqs, 1);
  dim3 block(NUM_THREADS);
  switch (head_size) {
    // NOTE(woosuk): To reduce the compilation time, we only compile for the
    // head sizes that we use in the model. However, we can easily extend this
    // to support any head size which is a multiple of 16.
    case 32:
      LAUNCH_PAGED_ATTENTION_V1(32);
      break;
    case 64:
      LAUNCH_PAGED_ATTENTION_V1(64);
      break;
    case 80:
      LAUNCH_PAGED_ATTENTION_V1(80);
      break;
    case 96:
      LAUNCH_PAGED_ATTENTION_V1(96);
      break;
    case 112:
      LAUNCH_PAGED_ATTENTION_V1(112);
      break;
    case 120:
      LAUNCH_PAGED_ATTENTION_V1(120);
      break;
    case 128:
      LAUNCH_PAGED_ATTENTION_V1(128);
      break;
    case 192:
      LAUNCH_PAGED_ATTENTION_V1(192);
      break;
    case 256:
      LAUNCH_PAGED_ATTENTION_V1(256);
      break;
    default:
      printf("Error: unsupported head size: %d\n", head_size);
      break;
  }
}

template <typename T, int block_size = 16> 
void attention_page (int num_seqs, 
                     int num_query_heads,
                     int num_kv_heads,
                     int head_size,
                     int max_seq_len,
                     int num_blocks,
                     int repeat)
{
    printf("kv cache block size = %d\n", block_size);

    const unsigned long seed = 1234;
    const bool use_alibi = true;

    assert(num_query_heads % num_kv_heads == 0);
    //int num_queries_per_kv = num_query_heads / num_kv_heads;

    // note kscale and vscale are not used
    const float scale = 1.f / std::sqrt((float)head_size);

    // Allocate query
    const int query_elems = num_seqs * num_query_heads * head_size;
    const int q_stride = num_query_heads * head_size; // query.stride(0)

    T* query_d;
    GPU_CHECK(hipMalloc(&query_d, query_elems * sizeof(T)));

    T* query_h = (T*) malloc (query_elems * sizeof(T));

    const int threads = 256;
    int blocks = (query_elems + threads - 1) / threads;

    uniform_fill_kernel<T><<<blocks, threads>>>(query_d, query_elems, -scale, scale, seed);
    GPU_CHECK(hipMemcpy(query_h, query_d, sizeof(T) * query_elems, hipMemcpyDeviceToHost));

    // Allocate outputs
    T *out_d;
    GPU_CHECK(hipMalloc(&out_d, query_elems * sizeof(T)));

    T *out_h = (T*) malloc (query_elems * sizeof(T));
    T *out_r = (T*) malloc (query_elems * sizeof(T));

    // ALiBi slopes
    float* alibi_d = nullptr;
    float* alibi_h = nullptr;
    if (use_alibi) {
        GPU_CHECK(hipMalloc(&alibi_d, num_query_heads * sizeof(float)));
        alibi_h = (float*) malloc (num_query_heads * sizeof(float));

        blocks = (num_query_heads + threads - 1) / threads;
        norm_fill_kernel<float><<<blocks, threads>>>( alibi_d, num_query_heads, seed);
        GPU_CHECK(hipMemcpy(alibi_h, alibi_d, sizeof(float) * num_query_heads, hipMemcpyDeviceToHost));
    }

    // seq_lens
    int* seq_lens_h = (int*)malloc(num_seqs * sizeof(int));

    srand(seed);
    for (int i = 0; i < num_seqs; i++) {
        seq_lens_h[i] = rand() % max_seq_len + 1; // variable length [1, max_seq_len]
    }

    int* seq_lens_d;
    GPU_CHECK(hipMalloc(&seq_lens_d, num_seqs * sizeof(int)));
    GPU_CHECK(hipMemcpy( seq_lens_d, seq_lens_h, num_seqs * sizeof(int), hipMemcpyHostToDevice));

    // Block tables
    const int max_num_blocks_per_seq = (max_seq_len + block_size - 1) / block_size;
    const int block_tables_size = num_seqs * max_num_blocks_per_seq;

    int* block_tables_h = (int*)malloc(block_tables_size * sizeof(int));

    for (int s = 0; s < num_seqs; s++) {
        for (int b = 0; b < max_num_blocks_per_seq; b++) {
            block_tables_h[s * max_num_blocks_per_seq + b] = rand() % num_blocks; // [0, num_blocks-1]
        }
    }

    int* block_tables_d;
    GPU_CHECK(hipMalloc(&block_tables_d, block_tables_size * sizeof(int)));
    GPU_CHECK(hipMemcpy(block_tables_d, block_tables_h, block_tables_size * sizeof(int), hipMemcpyHostToDevice));

    // KV cache allocation
    int num_layers = 1;
    int kv_block_stride, kv_head_stride;

    auto caches = create_kv_caches_with_random<T>(
      num_blocks,
      block_size,
      num_layers,
      num_kv_heads,
      head_size,
      seed,
      kv_block_stride,
      kv_head_stride
    );

    // device points
    auto key_cache_d = caches.key_caches[0];
    auto value_cache_d = caches.value_caches[0];

    int64_t kv_elems = (int64_t)num_blocks * num_kv_heads * head_size * block_size;
    T *key_cache_h = (T*) malloc (sizeof(T) * kv_elems);
    T *value_cache_h = (T*) malloc (sizeof(T) * kv_elems);
    GPU_CHECK(hipMemcpy(key_cache_h, key_cache_d, sizeof(T) * kv_elems, hipMemcpyDeviceToHost));
    GPU_CHECK(hipMemcpy(value_cache_h, value_cache_d, sizeof(T) * kv_elems, hipMemcpyDeviceToHost));

    GPU_CHECK(hipDeviceSynchronize());
    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
      paged_attention_v1_launcher<T, T, block_size, false>(
        out_d,
        query_d,
        key_cache_d,
        value_cache_d,
        num_kv_heads,
        scale,
        block_tables_d,
        seq_lens_d,
        max_seq_len,
        alibi_d,
        0, //const int tp_rank,
        0, 0, //const int blocksparse_local_blocks, const int blocksparse_vert_stride,
        0, 0, //const int blocksparse_block_size, const int blocksparse_head_sliding_step,
        num_seqs,
        num_kv_heads,
        head_size,
        max_num_blocks_per_seq,
        q_stride,
        kv_block_stride,
        kv_head_stride
      );
    }

    GPU_CHECK(hipDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of the kernel: %f (us)\n", (time * 1e-3f) / repeat);

    GPU_CHECK(hipMemcpy(out_h, out_d, sizeof(T) * query_elems, hipMemcpyDeviceToHost));

    //printf("Running PagedAttention CPU Reference...\n");
    PagedAttentionParams<T> params;
    params.out = out_r;
    params.query = query_h;
    params.key_cache = key_cache_h;
    params.value_cache = value_cache_h;
    params.block_tables = block_tables_h;
    params.seq_lens = seq_lens_h;
    params.alibi_slopes = alibi_h;
    params.num_seqs = num_seqs;
    params.num_query_heads = num_query_heads;
    params.num_kv_heads = num_kv_heads;
    params.head_size = head_size;
    params.block_size = block_size;
    params.max_num_blocks_per_seq = max_num_blocks_per_seq;
    params.scale = scale;

    reference(params);
   
    const float atol = 1e-3f, rtol = 1e-5f;
    bool ok = true;
    for (int i = 0; i < query_elems; i++) {
      if (std::fabs(float(out_h[i] - out_r[i])) > 
          atol + rtol * std::fabs((float)out_r[i])) {
        printf("Mismatch at index %d: %f %f\n", i, (float)out_h[i], (float)out_r[i]);
        ok = false;
        break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
    if (!ok) exit(1);

    free(query_h);
    free(out_h);
    free(out_r);
    free(seq_lens_h);
    free(block_tables_h);
    free(key_cache_h);
    free(value_cache_h);
    GPU_CHECK(hipFree(query_d));
    GPU_CHECK(hipFree(seq_lens_d));
    GPU_CHECK(hipFree(block_tables_d));
    if (alibi_d) {
      GPU_CHECK(hipFree(alibi_d));
      free(alibi_h);
    }

    for (int l = 0; l < num_layers; ++l) {
      GPU_CHECK(hipFree(caches.key_caches[l]));
      GPU_CHECK(hipFree(caches.value_caches[l]));
    }
    free(caches.key_caches);
    free(caches.value_caches);
}

int main(int argc, char* argv[])
{
   if (argc != 7) {
     printf("Usage: %s <batch size> <number of query heads> <head size> ", argv[0]);
     printf("<max_seq_len> <number of cache blocks> <repeat>\n");
     printf("head size choices [32, 64, 80, 96, 112, 120, 128, 192, 256]\n");
     return 1;
   }
    int num_seqs = atoi(argv[1]);        // batch size 8
    int num_query_heads = atoi(argv[2]); //32, 64
    int num_kv_heads = num_query_heads;  // there exist bugs in the vllm kernel when num_kv_heads and num_query_heads diff 
    int head_size = atoi(argv[3]);       // choices=[32, 64, 80, 96, 112, 120, 128, 192, 256],
    int max_seq_len = atoi(argv[4]);     // choices=[4096]
    int num_blocks = atoi(argv[5]);      //128*1024
    int repeat = atoi(argv[6]);

    printf("query shape = [%d (num_seqs), %d (num_query_heads), %d (head_size)]\n",
           num_seqs, num_query_heads, head_size);
    printf("Number of kv heads = %d\n", num_kv_heads);
    printf("Number of kv cache blocks = %d\n", num_blocks);
    printf("max_seq_len = %d\n", max_seq_len);
    
    //const int block_size = 16; // choices=[16, 32], a template parameter for paged_attention_v1_launcher
                               // 32 may cause OOM in FP32 kvcache allocation

    
    printf("\n-------------------\nFP32 PageAttention v1\n--------------------\n");
    attention_page<float, 16>(num_seqs, num_query_heads,
      num_kv_heads, head_size, max_seq_len, num_blocks, repeat);

    printf("\n-------------------\nBF16 PageAttention v1\n--------------------\n");
    attention_page<__nv_bfloat16, 32>(num_seqs, num_query_heads,
      num_kv_heads, head_size, max_seq_len, num_blocks, repeat);

    return 0;
}
