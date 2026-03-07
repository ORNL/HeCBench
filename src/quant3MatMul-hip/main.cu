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

  hipMalloc(&d_input, sizeof(scalar_t) * vec_size);
  hipMemcpy(d_input, h_input, sizeof(scalar_t) * vec_size, hipMemcpyHostToDevice);

  hipMalloc(&d_input_half, sizeof(half) * vec_size);
  hipMemcpy(d_input_half, h_input_half, sizeof(half) * vec_size, hipMemcpyHostToDevice);

  hipMalloc(&d_weight, sizeof(int) * mat_size);
  hipMemcpy(d_weight, h_weight, sizeof(int) * mat_size, hipMemcpyHostToDevice);

  hipMalloc(&d_scale, sizeof(scalar_t) * mul_size);   
  hipMemcpy(d_scale, h_scale, sizeof(scalar_t) * mul_size, hipMemcpyHostToDevice);

  hipMalloc(&d_bias, sizeof(scalar_t) * mul_size);   
  hipMemcpy(d_bias, h_bias, sizeof(scalar_t) * mul_size, hipMemcpyHostToDevice);

  hipMalloc(&d_output, sizeof(scalar_t) * mul_size);

  // verify vecquant3matmul
  hipMemset(d_output, 0, sizeof(scalar_t) * mul_size);
  vecquant3matmul(d_input, d_weight, d_output, d_scale, d_bias,
                  mat_height, mat_width);
  hipMemcpy(h_output, d_output, sizeof(scalar_t) * mul_size, hipMemcpyDeviceToHost);
  bool ok = true;
  for (int i = 0; i < mul_size; i++) {
    if (h_output[i] != -49152.f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  // verify vecquant3matmul_faster
  hipMemset(d_output, 0, sizeof(scalar_t) * mul_size);
  vecquant3matmul_faster(d_input_half, d_weight, d_output, d_scale, d_bias,
                         mat_height, mat_width);
  hipMemcpy(h_output, d_output, sizeof(scalar_t) * mul_size, hipMemcpyDeviceToHost);
  ok = true;
  for (int i = 0; i < mul_size; i++) {
    if (h_output[i] != -49152.f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    hipMemset(d_output, 0, sizeof(scalar_t) * mul_size);
    vecquant3matmul(d_input, d_weight, d_output, d_scale, d_bias,
                    mat_height, mat_width);
  }
  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the kernel: %f (us)\n", (time * 1e-3f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    hipMemset(d_output, 0, sizeof(scalar_t) * mul_size);
    vecquant3matmul_faster(d_input_half, d_weight, d_output, d_scale, d_bias,
                           mat_height, mat_width);
  }
  hipDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the faster kernel: %f (us)\n", (time * 1e-3f) / repeat);

  hipFree(d_input);
  hipFree(d_input_half);
  hipFree(d_weight);
  hipFree(d_scale);
  hipFree(d_bias);
  hipFree(d_output);

  free(h_input);
  free(h_input_half);
  free(h_weight);
  free(h_scale);
  free(h_bias);
  free(h_output);
  return 0;
}
