#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <sycl/sycl.hpp>
#include "TensorInfo.h"
#include "reducer.h"

#define THREADS 256
#define BLOCKS(N) (N + THREADS - 1) / THREADS

template <typename scalar_t, ReductionType REDUCE>
void
scatter_kernel(const scalar_t *src_data,
               const TensorInfo<int64_t, int64_t> index_info,
               scalar_t *out_data, int E, int K, int N, int numel,
               const sycl::nd_item<3> &item) {

  int thread_idx = item.get_global_id(2);

  int b = thread_idx / (E * K);
  int k = thread_idx % K;

  if (thread_idx < numel) {
    int offset = IndexToOffset<int64_t, int64_t, -1>::get(
        thread_idx, index_info);
    int64_t idx = index_info.data[offset];

    Reducer<scalar_t, REDUCE>::atomic_write(out_data + b * N * K + idx * K + k,
                                            src_data[thread_idx]);
  }
}

template <typename scalar_t, ReductionType REDUCE>
void scatter(sycl::queue &q, int64_t num_elems, int repeat) {
  int64_t out_size = num_elems / 2;
  int64_t src_size_bytes = num_elems * sizeof(scalar_t);
  int64_t idx_size_bytes = num_elems * sizeof(int64_t);
  int64_t out_size_bytes = out_size * sizeof(scalar_t);

  int64_t *h_idx = (int64_t*) malloc (idx_size_bytes);
  scalar_t *h_src = (scalar_t*) malloc (src_size_bytes);
  scalar_t *h_out = (scalar_t*) malloc (out_size_bytes);
  scalar_t *h_ref = (scalar_t*) malloc (out_size_bytes);
  srand(123);
  for (int64_t i = 0; i < num_elems; i++) {
    h_idx[i] = rand() % out_size;
    h_src[i] = rand() % 2 ? 1 : -1;
  }

  scalar_t *d_src, *d_out;
  int64_t *d_idx;
  d_src = (scalar_t *)sycl::malloc_device(src_size_bytes, q);
  d_out = (scalar_t *)sycl::malloc_device(out_size_bytes, q);
  d_idx = (int64_t *)sycl::malloc_device(idx_size_bytes, q);
  q.memcpy(d_idx, h_idx, idx_size_bytes);
  q.memcpy(d_src, h_src, src_size_bytes);

  int64_t tensor_sizes[] = {num_elems};
  int64_t tensor_strides[] = {1};

  TensorInfo<int64_t, int64_t> index_info(d_idx, 1, tensor_sizes, tensor_strides);

  sycl::range<3> gws (1, 1, BLOCKS(num_elems) * THREADS);
  sycl::range<3> lws (1, 1, THREADS);

  int B = 1;
  int E = num_elems;
  int K = num_elems / (B * E);
  int N = num_elems / 2;

  for (int64_t i = 0; i < out_size; i++) {
    h_ref[i] = h_out[i] = Reducer<scalar_t, REDUCE>::init();
  }
  
  int64_t time = 0;
  for (int i = 0; i < repeat; i++) {
    q.memcpy(d_out, h_out, out_size_bytes).wait();
    auto start = std::chrono::steady_clock::now();
    q.parallel_for(sycl::nd_range<3>(gws, lws),
                     [=](sycl::nd_item<3> item) {
        scatter_kernel<scalar_t, REDUCE>(
           d_src, index_info, d_out, E, K, N, num_elems, item);
    }).wait();
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }
  printf("Average execution time of kernel: %f (us)\n", (time * 1e-3f) / repeat);

  q.memcpy(h_out, d_out, out_size_bytes).wait();

  for (int64_t i = 0; i < num_elems; i++) {
    Reducer<scalar_t, REDUCE>::update(h_ref[h_idx[i]], h_src[i]);
  }

  bool ok = true;
  for (int64_t i = 0; i < out_size; i++) {
    if ((double)h_out[i] - (double)h_ref[i] > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n\n", ok ? "PASS" : "FAIL");

  sycl::free(d_src, q);
  sycl::free(d_idx, q);
  sycl::free(d_out, q);
  free(h_src);
  free(h_idx);
  free(h_out);
  free(h_ref);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int64_t num_elements = atol(argv[1]);
  const int repeat = atoi(argv[2]);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  printf("INT32 scatter (mul, div, sum, min, max)\n"); 
  scatter<int32_t, MUL>(q, num_elements, repeat);
  scatter<int32_t, DIV>(q, num_elements, repeat);
  scatter<int32_t, SUM>(q, num_elements, repeat);
  scatter<int32_t, MIN>(q, num_elements, repeat);
  scatter<int32_t, MAX>(q, num_elements, repeat);
  printf("INT64 scatter (mul, div, sum, min, max)\n"); 
  scatter<int64_t, MUL>(q, num_elements, repeat);
  scatter<int64_t, DIV>(q, num_elements, repeat);
  scatter<int64_t, SUM>(q, num_elements, repeat);
  scatter<int64_t, MIN>(q, num_elements, repeat);
  scatter<int64_t, MAX>(q, num_elements, repeat);
  printf("FP32 scatter (mul, div, sum, min, max)\n"); 
  scatter<float, MUL>(q, num_elements, repeat);
  scatter<float, DIV>(q, num_elements, repeat);
  scatter<float, SUM>(q, num_elements, repeat);
  scatter<float, MIN>(q, num_elements, repeat);
  scatter<float, MAX>(q, num_elements, repeat);
  printf("FP64 scatter (mul, div, sum, min, max)\n"); 
  scatter<double, MUL>(q, num_elements, repeat);
  scatter<double, DIV>(q, num_elements, repeat);
  scatter<double, SUM>(q, num_elements, repeat);
  scatter<double, MIN>(q, num_elements, repeat);
  scatter<double, MAX>(q, num_elements, repeat);
  if (!ok) return 1;
  return 0;
}
