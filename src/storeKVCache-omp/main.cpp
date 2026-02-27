#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <omp.h>

template <typename T>
void store_kv_cache(int32_t repeat)
{
  uint32_t num_layers = 8; // H
  uint32_t cache_size = 2 * 1024 * 1024 / num_layers;
  const auto elem_size = sizeof(uint64_t);

  T *h_indices = (T*) malloc (cache_size * sizeof(T));

  // D is item_size
  for (int32_t item_size = 2; item_size <= 1024; item_size *= 2) {

    uint64_t kvc_size = (uint64_t)num_layers * cache_size * item_size;
    uint64_t *h_k_cache = (uint64_t*) malloc (kvc_size * elem_size);
    uint64_t *h_v_cache = (uint64_t*) malloc (kvc_size * elem_size);
    uint64_t *r_k_cache = (uint64_t*) malloc (kvc_size * elem_size);
    uint64_t *r_v_cache = (uint64_t*) malloc (kvc_size * elem_size);

    #pragma omp target data map (alloc: h_k_cache[0:kvc_size], \
                                        h_v_cache[0:kvc_size])
    {
                                        
    for (uint32_t batch_size = 1; batch_size <= 16384; batch_size *= 2) {

      if (cache_size < batch_size) {
        printf("Warning: skip the test when cache size < batch_size\n");
        continue;
      }

      uint64_t kv_size = (uint64_t)num_layers * batch_size * item_size;
      uint64_t *h_k = (uint64_t*) malloc (kv_size * elem_size);
      uint64_t *h_v = (uint64_t*) malloc (kv_size * elem_size);

      memset(h_k_cache, 0, elem_size * kvc_size);
      memset(h_v_cache, 0, elem_size * kvc_size);
      #pragma omp target update to (h_k_cache[0:kvc_size])
      #pragma omp target update to (h_v_cache[0:kvc_size])

      #pragma omp target data map(alloc: h_k[0:kv_size], \
                                         h_v[0:kv_size], \
                                         h_indices[0:batch_size])
      {
        memset(r_k_cache, 0, elem_size * kvc_size);
        memset(r_v_cache, 0, elem_size * kvc_size);

        uint64_t hxd_size = num_layers * item_size;
        const auto row_size_bytes = elem_size * hxd_size;
        const auto kv_cache_stride_bytes = row_size_bytes;
        const auto kv_input_stride_bytes = row_size_bytes;
        const auto kv_cache_stride = kv_cache_stride_bytes / elem_size;
        const auto kv_input_stride = kv_input_stride_bytes / elem_size;

        // random permutate indices on host..
        std::mt19937 gen{123};
        std::iota(h_indices, h_indices + cache_size, 0);
        std::shuffle(h_indices, h_indices + cache_size, gen);
        #pragma omp target update to (h_indices[0:batch_size])

        std::normal_distribution<float> d{0.f, 1.f};
        for (uint64_t i = 0; i < kv_size; i++) {
          h_k[i] = d(gen) * kv_size;
          h_v[i] = d(gen) * kv_size;
        }
        // reference
        #pragma omp parallel for collapse(2)
        for (uint32_t i = 0; i < batch_size; i++) {
          for (uint64_t j = 0; j < hxd_size; j++) {
            r_k_cache[h_indices[i] * kv_cache_stride + j] = h_k[i * kv_input_stride + j];
            r_v_cache[h_indices[i] * kv_cache_stride + j] = h_v[i * kv_input_stride + j];
          }
        }

        #pragma omp target update to (h_k[0:kv_size])
        #pragma omp target update to (h_v[0:kv_size])

        //const auto num_threads = 256;

        printf("item size %4d, batch size %6d : ", item_size, batch_size);
        auto start = std::chrono::steady_clock::now();

        for (int32_t i = 0; i < repeat; i++) {
          #pragma omp target teams distribute parallel for collapse(2)
          for (uint32_t i = 0; i < batch_size; i++) {
            for (uint64_t j = 0; j < hxd_size; j++) {
              h_k_cache[h_indices[i] * kv_cache_stride + j] = h_k[i * kv_input_stride + j];
              h_v_cache[h_indices[i] * kv_cache_stride + j] = h_v[i * kv_input_stride + j];
            }
          }
        }

        auto end = std::chrono::steady_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        printf("Average execution time of store cache kernel: %f (us)\n", (time * 1e-3f) / repeat);

        #pragma omp target update from (h_k_cache[0:kvc_size])
        #pragma omp target update from (h_v_cache[0:kvc_size])

        bool ok = true;
        #ifdef DEBUG
        for (int32_t i = 0; i < cache_size; i++) {
          for (uint64_t j = 0; j < hxd_size; j++) {
            if (r_k_cache[i* hxd_size + j] != h_k_cache[i* hxd_size + j]) {
              printf("Error: kcache mismatch @ batch=%d loc=%lu %lu != %lu\n",
                     i, j, r_k_cache[i* hxd_size + j], h_k_cache[i* hxd_size + j]);
              ok = false;
              break;
            }
            if (r_v_cache[i* hxd_size + j] != h_v_cache[i* hxd_size + j]) {
              printf("Error: vcache mismatch @ batch=%d loc=%lu %lu != %lu\n",
                     i, j, r_v_cache[i* hxd_size + j], h_v_cache[i* hxd_size + j]);
              ok = false;
              break;
            }
          }
        }
        #else
        ok = (memcmp(r_k_cache, h_k_cache, elem_size * kvc_size) == 0) &&
             (memcmp(r_v_cache, h_v_cache, elem_size * kvc_size) == 0);
        #endif
        printf("%s\n", ok ? "PASS" : "FAIL");

        free(h_k);
        free(h_v);
      }
    }
    free(h_k_cache);
    free(h_v_cache);
    free(r_k_cache);
    free(r_v_cache);
  }}
  free(h_indices);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  store_kv_cache<uint64_t>(repeat);
}
