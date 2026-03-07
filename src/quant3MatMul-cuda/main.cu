#include <chrono>
#include <cstdio>
#include <cstdlib>
#include "kernels.h"

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);

  // Reference https://github.com/IST-DASLab/gptq/blob/main/test_kernel.py
  // Benchmarking OPT-175B FC2 matvec 
  const int m = 12288 * 4;
  const int n = 12288;
  const int mat_height = (m / 1024 * 96);  // 4608
  const int mat_width = n;
  const int mat_size = mat_height * mat_width;
  const int vec_size = 1 * m;
  const int mul_size = 1 * n;

  using scalar_t = float;

  // initialize the vectors 
  scalar_t *h_input = (scalar_t*) malloc (sizeof(scalar_t) * vec_size);
  half *h_input_half = (half*) malloc (sizeof(half) * vec_size);
  for (int i = 0; i < vec_size; i++) {
    h_input_half[i] = h_input[i] = 1.f;
  }

  // initialize the weights 
  int *h_weight = (int*) malloc (sizeof(int) * mat_size);
  int p1 = 0b01001001001001001001001001001001;
  for (int i = 0; i < mat_height; i += 3) {
    for (int j = 0; j < mat_width; j++) {
      h_weight[i * mat_width + j] = p1;
      h_weight[(i+1) * mat_width + j] = p1 << 1;
      h_weight[(i+2) * mat_width + j] = p1 << 2;
    }
  }

  scalar_t *h_scale = (scalar_t*) malloc (sizeof(scalar_t) * mul_size);
  for (int i = 0; i < mul_size; i++) {
    h_scale[i] = -1.f;
  }

  scalar_t *h_bias = (scalar_t*) malloc (sizeof(scalar_t) * mul_size);
  for (int i = 0; i < mul_size; i++) {
    h_bias[i] = 0.f;
  }

  scalar_t *h_output = (scalar_t*) malloc (sizeof(scalar_t) * mul_size);
  
  scalar_t *d_input;
  half *d_input_half;
  int *d_weight;
  scalar_t *d_scale, *d_bias, *d_output;

  cudaMalloc(&d_input, sizeof(scalar_t) * vec_size);
  cudaMemcpy(d_input, h_input, sizeof(scalar_t) * vec_size, cudaMemcpyHostToDevice);

  cudaMalloc(&d_input_half, sizeof(half) * vec_size);
  cudaMemcpy(d_input_half, h_input_half, sizeof(half) * vec_size, cudaMemcpyHostToDevice);

  cudaMalloc(&d_weight, sizeof(int) * mat_size);
  cudaMemcpy(d_weight, h_weight, sizeof(int) * mat_size, cudaMemcpyHostToDevice);

  cudaMalloc(&d_scale, sizeof(scalar_t) * mul_size);   
  cudaMemcpy(d_scale, h_scale, sizeof(scalar_t) * mul_size, cudaMemcpyHostToDevice);

  cudaMalloc(&d_bias, sizeof(scalar_t) * mul_size);   
  cudaMemcpy(d_bias, h_bias, sizeof(scalar_t) * mul_size, cudaMemcpyHostToDevice);

  cudaMalloc(&d_output, sizeof(scalar_t) * mul_size);

  // verify vecquant3matmul
  cudaMemset(d_output, 0, sizeof(scalar_t) * mul_size);
  vecquant3matmul(d_input, d_weight, d_output, d_scale, d_bias,
                  mat_height, mat_width);
  cudaMemcpy(h_output, d_output, sizeof(scalar_t) * mul_size, cudaMemcpyDeviceToHost);
  bool ok = true;
  for (int i = 0; i < mul_size; i++) {
    if (h_output[i] != -49152.f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  // verify vecquant3matmul_faster
  cudaMemset(d_output, 0, sizeof(scalar_t) * mul_size);
  vecquant3matmul_faster(d_input_half, d_weight, d_output, d_scale, d_bias,
                         mat_height, mat_width);
  cudaMemcpy(h_output, d_output, sizeof(scalar_t) * mul_size, cudaMemcpyDeviceToHost);
  ok = true;
  for (int i = 0; i < mul_size; i++) {
    if (h_output[i] != -49152.f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    cudaMemset(d_output, 0, sizeof(scalar_t) * mul_size);
    vecquant3matmul(d_input, d_weight, d_output, d_scale, d_bias,
                    mat_height, mat_width);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the kernel: %f (us)\n", (time * 1e-3f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    cudaMemset(d_output, 0, sizeof(scalar_t) * mul_size);
    vecquant3matmul_faster(d_input_half, d_weight, d_output, d_scale, d_bias,
                           mat_height, mat_width);
  }
  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the faster kernel: %f (us)\n", (time * 1e-3f) / repeat);

  cudaFree(d_input);
  cudaFree(d_input_half);
  cudaFree(d_weight);
  cudaFree(d_scale);
  cudaFree(d_bias);
  cudaFree(d_output);

  free(h_input);
  free(h_input_half);
  free(h_weight);
  free(h_scale);
  free(h_bias);
  free(h_output);
  if (!ok) return 1;
  if (!ok) return 1;
  return 0;
}
