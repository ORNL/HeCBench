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

  using half = sycl::half;
  using scalar_t = float;

  // initialize the vectors 
  scalar_t *h_input = (scalar_t*) malloc (sizeof(scalar_t) * vec_size);
  half *h_input_half = (half *)malloc(sizeof(half) * vec_size);
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
  
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  scalar_t *d_input;
  half *d_input_half;
  int *d_weight;
  scalar_t *d_scale, *d_bias, *d_output;

  d_input = sycl::malloc_device<scalar_t>(vec_size, q);
  q.memcpy(d_input, h_input, sizeof(scalar_t) * vec_size);

  d_input_half = sycl::malloc_device<half>(vec_size, q);
  q.memcpy(d_input_half, h_input_half, sizeof(half) * vec_size);

  d_weight = sycl::malloc_device<int>(mat_size, q);
  q.memcpy(d_weight, h_weight, sizeof(int) * mat_size);

  d_scale = sycl::malloc_device<scalar_t>(mul_size, q);
  q.memcpy(d_scale, h_scale, sizeof(scalar_t) * mul_size);

  d_bias = sycl::malloc_device<scalar_t>(mul_size, q);
  q.memcpy(d_bias, h_bias, sizeof(scalar_t) * mul_size);

  d_output = sycl::malloc_device<scalar_t>(mul_size, q);

  // verify vecquant3matmul
  q.memset(d_output, 0, sizeof(scalar_t) * mul_size);
  vecquant3matmul(q, d_input, d_weight, d_output, d_scale, d_bias,
                  mat_height, mat_width);
  q.memcpy(h_output, d_output, sizeof(scalar_t) * mul_size).wait();
  bool ok = true;
  for (int i = 0; i < mul_size; i++) {
    if (h_output[i] != -49152.f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  // verify vecquant3matmul_faster
  q.memset(d_output, 0, sizeof(scalar_t) * mul_size);
  vecquant3matmul_faster(q, d_input_half, d_weight, d_output, d_scale, d_bias,
                         mat_height, mat_width);
  q.memcpy(h_output, d_output, sizeof(scalar_t) * mul_size).wait();
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
    q.memset(d_output, 0, sizeof(scalar_t) * mul_size);
    vecquant3matmul(q, d_input, d_weight, d_output, d_scale, d_bias,
                    mat_height, mat_width);
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the kernel: %f (us)\n", (time * 1e-3f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.memset(d_output, 0, sizeof(scalar_t) * mul_size);
    vecquant3matmul_faster(q, d_input_half, d_weight, d_output, d_scale, d_bias,
                           mat_height, mat_width);
  }
  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the faster kernel: %f (us)\n", (time * 1e-3f) / repeat);

  sycl::free(d_input, q);
  sycl::free(d_input_half, q);
  sycl::free(d_weight, q);
  sycl::free(d_scale, q);
  sycl::free(d_bias, q);
  sycl::free(d_output, q);

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
