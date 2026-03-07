#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <random>
#include <hip/hip_runtime.h>
#include "kernels.h"
#include "reference.h"

inline int grids(int p, int b, int warp_size) {
  int waves = b / warp_size;
  return (p + waves - 1) / waves;
}

float* attention_device(const float* key, const float* value, const float* query,
                        const int n, const int d, const int impl_num, const int repeat)
{
  size_t q_size = (size_t)d;
  size_t r_size = (size_t)n;
  size_t kv_size = (size_t)d * n;
  size_t r_size_bytes = r_size * sizeof(float);
  size_t q_size_bytes = q_size * sizeof(float);
  size_t kv_size_bytes = kv_size * sizeof(float);

  // input
  float *d_key;
  hipMalloc((void**)&d_key, kv_size_bytes);
  hipMemcpy(d_key, key, kv_size_bytes, hipMemcpyHostToDevice);

  float *d_value;
  hipMalloc((void**)&d_value, kv_size_bytes);
  hipMemcpy(d_value, value, kv_size_bytes, hipMemcpyHostToDevice);

  float *d_query;
  hipMalloc((void**)&d_query, q_size_bytes);
  hipMemcpy(d_query, query, q_size_bytes, hipMemcpyHostToDevice);

  // intermediate
  float *d_dot_product;
  hipMalloc((void**)&d_dot_product, r_size_bytes);

  float *d_exp_sum;
  hipMalloc((void**)&d_exp_sum, sizeof(float));

  // result
  float *output = (float*) malloc (q_size_bytes);
  float *d_output;
  hipMalloc((void**)&d_output, q_size_bytes);

  int warp_size;
  hipDeviceGetAttribute(&warp_size, hipDeviceAttributeWarpSize, 0);

  hipDeviceSynchronize();

  if (impl_num == 3) {

    int num_blocks = grids(n, 256, warp_size);

    auto start = std::chrono::steady_clock::now();

    for (int k = 0; k < repeat; k++) {
      hipMemset(d_exp_sum, 0, 4);
      attention_kernel1_warpReduce<<<num_blocks, 256>>>(d_key, d_query, d_dot_product, d_exp_sum, n, d);
      attention_kernel2_blockReduce<<<d, 256>>>(d_exp_sum, d_dot_product, d_value, d_output, n, d);
    }

    hipDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernels %f (ms)\n", time * 1e-6f / repeat);
  }

  else if (impl_num == 2) {

    int num_blocks = grids(n, 256, warp_size);
    int num_blocks2 = grids(d, 256, warp_size);

    auto start = std::chrono::steady_clock::now();

    for (int k = 0; k < repeat; k++) {
      hipMemset(d_exp_sum, 0, 4);
      attention_kernel1_warpReduce<<<num_blocks, 256>>>(d_key, d_query, d_dot_product, d_exp_sum, n, d);
      attention_kernel2_warpReduce<<<num_blocks2, 256>>>(d_exp_sum, d_dot_product, d_value, d_output, n, d);
    }

    hipDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernels %f (ms)\n", time * 1e-6f / repeat);
  }

  else if (impl_num == 1) {

    auto start = std::chrono::steady_clock::now();

    for (int k = 0; k < repeat; k++) {
      hipMemset(d_exp_sum, 0, 4);
      attention_kernel1_blockReduce<<<n, 256>>>(d_key, d_query, d_dot_product, d_exp_sum, n, d);
      attention_kernel2_blockReduce<<<d, 256>>>(d_exp_sum, d_dot_product, d_value, d_output, n, d);
    }

    hipDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernels %f (ms)\n", time * 1e-6f / repeat);
  }

  else {
    float *d_score;
    hipMalloc((void**)&d_score, r_size_bytes);

    auto start = std::chrono::steady_clock::now();

    for (int k = 0; k < repeat; k++) {
      hipMemset(d_exp_sum, 0, 4);
      attention_kernel1<<<(n+255)/256, 256>>>(d_key, d_query, d_dot_product, d_exp_sum, n, d);
      attention_kernel2<<<(n+255)/256, 256>>>(d_exp_sum, d_dot_product, d_score, n);
      attention_kernel3<<<(d+255)/256, 256>>>(d_score, d_value, d_output, n, d);
    }

    hipDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernels %f (ms)\n", time * 1e-6f / repeat);
    hipFree(d_score);
  }

  hipMemcpy(output, d_output, q_size_bytes, hipMemcpyDeviceToHost);
  hipFree(d_value);
  hipFree(d_output);
  hipFree(d_key);
  hipFree(d_dot_product);
  hipFree(d_exp_sum);
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
