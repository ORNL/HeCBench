#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <numeric>
#include <random>
#include <sycl/sycl.hpp>
#include "kernels.h"

template <typename T> void store_kv_cache(sycl::queue &q, int32_t repeat) try {
   uint32_t num_layers = 8; // H
   uint32_t cache_size = 2 * 1024 * 1024 / num_layers;
   const auto elem_size = sizeof(uint64_t);

   uint64_t *k, *v, *k_cache, *v_cache;
   T *indices;

   T *h_indices = (T*) malloc (cache_size * sizeof(T));

   auto sg_sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
   auto r = std::max_element(sg_sizes.begin(), sg_sizes.end());
   int warpSize = *r;

   // D is item_size
   for (int32_t item_size = warpSize/2/num_layers; item_size <= 1024; item_size *= 2) {

     uint64_t kvc_size = (uint64_t)num_layers * cache_size * item_size;
     uint64_t *h_k_cache = (uint64_t*) malloc (kvc_size * elem_size);
     uint64_t *h_v_cache = (uint64_t*) malloc (kvc_size * elem_size);
     uint64_t *r_k_cache = (uint64_t*) malloc (kvc_size * elem_size);
     uint64_t *r_v_cache = (uint64_t*) malloc (kvc_size * elem_size);

     k_cache = (uint64_t *)sycl::malloc_device(kvc_size * elem_size, q);
     v_cache = (uint64_t *)sycl::malloc_device(kvc_size * elem_size, q);

     for (uint32_t batch_size = 1; batch_size <= 16384; batch_size *= 2) {

       if (cache_size < batch_size) {
         printf("Warning: skip the test when cache size < batch_size\n");
         continue;
       }

       uint64_t kv_size = (uint64_t)num_layers * batch_size * item_size;
       uint64_t *h_k = (uint64_t*) malloc (kv_size * elem_size);
       uint64_t *h_v = (uint64_t*) malloc (kv_size * elem_size);

       k = (uint64_t *)sycl::malloc_device(kv_size * elem_size, q);
       v = (uint64_t *)sycl::malloc_device(kv_size * elem_size, q);
       indices = (T *)sycl::malloc_device(batch_size * sizeof(T), q);

       q.memset(k_cache, 0, elem_size * kvc_size).wait();
       q.memset(v_cache, 0, elem_size * kvc_size).wait();
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
       q.memcpy(indices, h_indices, sizeof(T) * batch_size).wait();

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

       q.memcpy(k, h_k, elem_size * kv_size).wait();
       q.memcpy(v, h_v, elem_size * kv_size).wait();

       const auto num_threads = 256;
       const auto num_warps = num_threads / warpSize;
       auto num_blocks = (batch_size + num_warps - 1) / num_warps;

       printf("item size %4d, batch size %6d : ", item_size, batch_size);
       q.wait();
       auto start = std::chrono::steady_clock::now();

       for (int32_t i = 0; i < repeat; i++) {
         sycl::nd_range<1> nr (sycl::range<1>(num_blocks * num_threads),
                               sycl::range<1>(num_threads));
         if (row_size_bytes % (elem_size * warpSize * 4) == 0) {
           const auto items_per_warp = row_size_bytes / (elem_size * warpSize * 4);
           if (warpSize == 64)  
             q.parallel_for(nr, [=](sycl::nd_item<1> item) {
                  store_kv_cache_256x1_v4<uint64_t, 64>(k_cache, v_cache, indices, batch_size,
                                          k, v, kv_cache_stride, kv_input_stride,
                                          items_per_warp, item);
                });
           else
             q.parallel_for(nr, [=](sycl::nd_item<1> item) {
                  store_kv_cache_256x1_v4(k_cache, v_cache, indices, batch_size,
                                          k, v, kv_cache_stride, kv_input_stride,
                                          items_per_warp, item);
                });
         }
         else if (row_size_bytes % (elem_size * warpSize * 2) == 0) {
           const auto items_per_warp = row_size_bytes / (elem_size * warpSize * 2);
           if (warpSize == 64)  
             q.parallel_for(nr, [=](sycl::nd_item<1> item) {
                store_kv_cache_256x1_v2<uint64_t, 64>(k_cache, v_cache, indices, batch_size,
                                        k, v, kv_cache_stride, kv_input_stride,
                                        items_per_warp, item);
              });
           else
             q.parallel_for(nr, [=](sycl::nd_item<1> item) {
                store_kv_cache_256x1_v2(k_cache, v_cache, indices, batch_size,
                                        k, v, kv_cache_stride, kv_input_stride,
                                        items_per_warp, item);
              });
         }
         else if (row_size_bytes % (elem_size * warpSize) == 0) {
           const auto items_per_warp = row_size_bytes / (elem_size * warpSize);
           if (warpSize == 64)  
             q.parallel_for(nr, [=](sycl::nd_item<1> item) {
                store_kv_cache_256x1<uint64_t, 64>(k_cache, v_cache, indices, batch_size, k,
                                     v, kv_cache_stride, kv_input_stride,
                                     items_per_warp, item);
              });
           else
             q.parallel_for(nr, [=](sycl::nd_item<1> item) {
                store_kv_cache_256x1(k_cache, v_cache, indices, batch_size, k,
                                     v, kv_cache_stride, kv_input_stride,
                                     items_per_warp, item);
              });
         } else if (row_size_bytes % (elem_size * warpSize / 2) == 0) {
           const auto items_per_warp = row_size_bytes / (elem_size * warpSize / 2);
           if (warpSize == 64)  
             q.parallel_for(nr, [=](sycl::nd_item<1> item) {
                store_kv_cache_128x2<uint64_t, 64>(k_cache, v_cache, indices, batch_size, k,
                                     v, kv_cache_stride, kv_input_stride,
                                     items_per_warp, item);
              });
           else
             q.parallel_for(nr, [=](sycl::nd_item<1> item) {
                store_kv_cache_128x2(k_cache, v_cache, indices, batch_size, k,
                                     v, kv_cache_stride, kv_input_stride,
                                     items_per_warp, item);
              });
         } else {
           printf("The last dimension size bytes of k and v must be divisible by %lu at least, got: %lu\n",
                  elem_size * warpSize / 2, row_size_bytes);
           break;
         }
       }

       q.wait();
       auto end = std::chrono::steady_clock::now();
       auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
       printf("Average execution time of store cache kernel: %f (us)\n", (time * 1e-3f) / repeat);

       q.memcpy(h_k_cache, k_cache, elem_size * kvc_size).wait();
       q.memcpy(h_v_cache, v_cache, elem_size * kvc_size).wait();

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

       sycl::free(k, q);
       sycl::free(v, q);
       sycl::free(indices, q);
       free(h_k);
       free(h_v);
     }
     sycl::free(k_cache, q);
     sycl::free(v_cache, q);
     free(h_k_cache);
     free(h_v_cache);
     free(r_k_cache);
     free(r_v_cache);
   }
   free(h_indices);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  store_kv_cache<uint64_t>(q, repeat);
}
