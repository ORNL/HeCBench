#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <random>
#include <sycl/sycl.hpp>
#include "kernels.h"
#include "reference.h"

inline int grids(int p, int b, int warpSize) {
  int waves = b / warpSize;
  return (p + waves - 1) / waves;
}

float* attention_device(const float* key, const float* value, const float* query,
                        const int n, const int d, const int impl_num, const int repeat)
{
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  auto sg_sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
  auto r = std::max_element(sg_sizes.begin(), sg_sizes.end());
  int warpSize = *r;

  size_t q_size = (size_t)d;
  size_t r_size = (size_t)n;
  size_t kv_size = (size_t)d * n;
  size_t r_size_bytes = r_size * sizeof(float);
  size_t q_size_bytes = q_size * sizeof(float);
  size_t kv_size_bytes = kv_size * sizeof(float);

  // input
  float *d_key;
  d_key = sycl::malloc_device<float>(kv_size, q);
  q.memcpy(d_key, key, kv_size_bytes).wait();

  float *d_value;
  d_value = sycl::malloc_device<float>(kv_size, q);
  q.memcpy(d_value, value, kv_size_bytes).wait();

  float *d_query;
  d_query = sycl::malloc_device<float>(d, q);
  q.memcpy(d_query, query, q_size_bytes).wait();

  // intermediate
  float *d_dot_product;
  d_dot_product = sycl::malloc_device<float>(n, q);

  float *d_exp_sum;
  d_exp_sum = sycl::malloc_device<float>(1, q);

  // result
  float *output = (float*) malloc (q_size_bytes);
  float *d_output;
  d_output = sycl::malloc_device<float>(d, q);

  q.wait();

  if (impl_num == 3) {

    auto start = std::chrono::steady_clock::now();

    for (int k = 0; k < repeat; k++) {
      q.memset(d_exp_sum, 0, 4);
      q.parallel_for(sycl::nd_range<1>(sycl::range<1>(grids(n, 256, warpSize) * 256),
                                       sycl::range<1>(256)),
                         [=](sycl::nd_item<1> item) {
                           attention_kernel1_warpReduce(d_key, d_query, d_dot_product,
                                              d_exp_sum, n, d, item);
                         });
      q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(d * 256),
                                           sycl::range<1>(256)),
                         [=](sycl::nd_item<1> item) {
                           attention_kernel2_blockReduce(d_exp_sum, d_dot_product,
                                               d_value, d_output, n, d,
                                               item);
                         });
      });
    }

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernels %f (ms)\n", time * 1e-6f / repeat);
  }

  else if (impl_num == 2) {

    auto start = std::chrono::steady_clock::now();

    for (int k = 0; k < repeat; k++) {
      q.memset(d_exp_sum, 0, 4);
      q.parallel_for(sycl::nd_range<1>(sycl::range<1>(grids(n, 256, warpSize) * 256),
                                       sycl::range<1>(256)),
                         [=](sycl::nd_item<1> item) {
                           attention_kernel1_warpReduce(d_key, d_query, d_dot_product,
                                              d_exp_sum, n, d, item);
                         });
      q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(grids(d, 256, warpSize) * 256),
                                           sycl::range<1>(256)),
                         [=](sycl::nd_item<1> item) {
                           attention_kernel2_warpReduce(d_exp_sum, d_dot_product,
                                               d_value, d_output, n, d,
                                               item);
                         });
      });
    }

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernels %f (ms)\n", time * 1e-6f / repeat);
  }

  else if (impl_num == 1) {

    auto start = std::chrono::steady_clock::now();

    for (int k = 0; k < repeat; k++) {
      q.memset(d_exp_sum, 0, 4);
      q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(n * 256),
                                           sycl::range<1>(256)),
                         [=](sycl::nd_item<1> item) {
                           attention_kernel1_blockReduce(d_key, d_query, d_dot_product,
                                               d_exp_sum, n, d, item);
                         });
      });
      q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(d * 256),
                                           sycl::range<1>(256)),
                         [=](sycl::nd_item<1> item) {
                           attention_kernel2_blockReduce(d_exp_sum, d_dot_product,
                                               d_value, d_output, n, d,
                                               item);
                         });
      });
    }

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernels %f (ms)\n", time * 1e-6f / repeat);
  }

  else {
    float *d_score;
    d_score = sycl::malloc_device<float>(n, q);

    auto start = std::chrono::steady_clock::now();

    for (int k = 0; k < repeat; k++) {
      q.memset(d_exp_sum, 0, 4);
      q.parallel_for(
          sycl::nd_range<1>(sycl::range<1>((n + 255) / 256) *
                            sycl::range<1>(256),
                            sycl::range<1>(256)),
          [=](sycl::nd_item<1> item) {
            attention_kernel1(d_key, d_query, d_dot_product, d_exp_sum, n, d, item);
          });
      q.parallel_for(
          sycl::nd_range<1>(sycl::range<1>((n + 255) / 256) *
                            sycl::range<1>(256),
                            sycl::range<1>(256)),
          [=](sycl::nd_item<1> item) {
            attention_kernel2(d_exp_sum, d_dot_product, d_score, n, item);
          });
      q.parallel_for(
          sycl::nd_range<1>(sycl::range<1>((d + 255) / 256) *
                            sycl::range<1>(256),
                            sycl::range<1>(256)),
          [=](sycl::nd_item<1> item) {
            attention_kernel3(d_score, d_value, d_output, n, d, item);
          });
    }

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernels %f (ms)\n", time * 1e-6f / repeat);
    sycl::free(d_score, q);
  }

  q.memcpy(output, d_output, q_size_bytes).wait();
  sycl::free(d_value, q);
  sycl::free(d_output, q);
  sycl::free(d_key, q);
  sycl::free(d_dot_product, q);
  sycl::free(d_exp_sum, q);
  return output;
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <rows> <columns> <implementation> <repeat>\n", argv[0]);
    printf("implementation 0: naive\n");
    printf("implementation 1: fused kernels with block reduce\n");
    printf("implementation 2: fused kernels with warp reduce\n");
    printf("implementation 3: fused kernels with mixed reduce\n");
    return 1;
  }
  const int n = atoi(argv[1]);
  const int d = atoi(argv[2]);
  const int k = atoi(argv[3]);
  const int r = atoi(argv[4]);

  // input
  size_t q_size = (size_t)d;
  size_t kv_size = (size_t)d * n;
  size_t q_size_bytes = q_size * sizeof(float);
  size_t kv_size_bytes = kv_size * sizeof(float);

  float* key = (float*) malloc (kv_size_bytes);
  float* value = (float*) malloc (kv_size_bytes);
  float* query = (float*) malloc (q_size_bytes);

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dist(-0.01f, 0.01f);

  for (size_t i = 0; i < kv_size; i++) {
    key[i] = dist(gen);
    value[i] = dist(gen);
    query[i % d] = dist(gen);
  }

  float* hout = attention_host(key, value, query, n, d);
  float* dout = attention_device(key, value, query, n, d, k, r);

  bool ok = true;
  for (int i = 0; i < d; i++) {
    if (fabsf(hout[i] - dout[i]) > 1e-3f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  free(key);
  free(value);
  free(query);
  free(dout);
  free(hout);
  return 0;
}
