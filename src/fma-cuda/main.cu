#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "kernels.h"
#include "reference.h"

template<typename scalar_t>
void fma(int NA, int NC, int C, int num_ops, int repeat) {
  size_t a_size = (size_t)NA * C;
  size_t b_size = C;
  size_t c_size = (size_t)NC * C;
  size_t a_size_bytes = a_size * sizeof(scalar_t);
  size_t b_size_bytes = b_size * sizeof(scalar_t);
  size_t c_size_bytes = c_size * sizeof(scalar_t);
  size_t idx_size_bytes = num_ops * sizeof(int);

  // use case
  // grad_in_features[in_map] += grad_output[out_map] * weight[i]
  scalar_t *h_a = (scalar_t*) malloc (a_size_bytes);
  scalar_t *h_b = (scalar_t*) malloc (b_size_bytes);
  scalar_t *h_c = (scalar_t*) malloc (c_size_bytes);
  int *h_i_idx = (int*) malloc (idx_size_bytes);
  int *h_o_idx = (int*) malloc (idx_size_bytes);
  scalar_t *h_r = (scalar_t*) malloc (c_size_bytes);

  srand(123);
  for (size_t i = 0; i < a_size; i++) {
    h_a[i] = rand() % 5;
  }
  for (size_t i = 0; i < b_size; i++) {
    h_b[i] = rand() % 3 - 1;
  }
  for (int i = 0; i < num_ops; i++) {
    h_i_idx[i] = rand() % NA;
  }
  // unique values (num_ops <= NC)
  for (int i = 0; i < num_ops; i++) {
    h_o_idx[i] = i;
  }

  scalar_t *d_a, *d_b, *d_c;
  int *d_i_idx, *d_o_idx;
  CHECK_CUDA( cudaMalloc(&d_a, a_size_bytes) )
  CHECK_CUDA( cudaMalloc(&d_b, b_size_bytes) )
  CHECK_CUDA( cudaMalloc(&d_c, c_size_bytes) )
  CHECK_CUDA( cudaMalloc(&d_i_idx, idx_size_bytes) )
  CHECK_CUDA( cudaMalloc(&d_o_idx, idx_size_bytes) )
  CHECK_CUDA( cudaMemcpy(d_i_idx, h_i_idx, idx_size_bytes, cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(d_o_idx, h_o_idx, idx_size_bytes, cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(d_a, h_a, a_size_bytes, cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(d_b, h_b, b_size_bytes, cudaMemcpyHostToDevice) )


  // warmup and verify
  CHECK_CUDA( cudaMemset(d_c, 0, c_size_bytes) )
  run_implicit_fma_templated<scalar_t, scalar_t, scalar_t>(
        d_a, d_b, d_c, d_i_idx, d_o_idx, num_ops, C, NA, NC, "basic");
  CHECK_CUDA( cudaMemcpy(h_c, d_c, c_size_bytes, cudaMemcpyDeviceToHost) )
  memset(h_r, 0, c_size_bytes);
  reference<scalar_t>(h_a, h_b, h_r, h_i_idx, h_o_idx, num_ops, C, NA, NC);
  bool ok = true;
  for (size_t i = 0; i < c_size; i++) {
    if ((double)h_c[i] - (double)h_r[i] > 1e-3) {
      ok = false;
      break;
    }
  }

  CHECK_CUDA( cudaMemset(d_c, 0, c_size_bytes) )
  run_implicit_fma_templated<scalar_t, scalar_t, scalar_t>(
        d_a, d_b, d_c, d_i_idx, d_o_idx, num_ops, C, NA, NC, "rowwise");
  CHECK_CUDA( cudaMemcpy(h_c, d_c, c_size_bytes, cudaMemcpyDeviceToHost) )
  for (size_t i = 0; i < c_size; i++) {
    if ((double)h_c[i] - (double)h_r[i] > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  double time = 0;
  for (int i = 0; i < repeat; i++) {
    CHECK_CUDA( cudaMemset(d_c, 0, c_size_bytes) )
    CHECK_CUDA( cudaDeviceSynchronize() )
    auto start = std::chrono::steady_clock::now();
    run_implicit_fma_templated<scalar_t, scalar_t, scalar_t>(
        d_a, d_b, d_c, d_i_idx, d_o_idx, num_ops, C, NA, NC, "basic");
    CHECK_CUDA( cudaDeviceSynchronize() )
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }
  printf("Average execution time of basic kernel: %f (us)\n", (time * 1e-3f) / repeat);

  time = 0;
  for (int i = 0; i < repeat; i++) {
    CHECK_CUDA( cudaMemset(d_c, 0, c_size_bytes) )
    CHECK_CUDA( cudaDeviceSynchronize() )
    auto start = std::chrono::steady_clock::now();
    run_implicit_fma_templated<scalar_t, scalar_t, scalar_t>(
        d_a, d_b, d_c, d_i_idx, d_o_idx, num_ops, C, NA, NC, "rowwise");
    CHECK_CUDA( cudaDeviceSynchronize() )
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }
  printf("Average execution time of rowwise kernel: %f (us)\n", (time * 1e-3f) / repeat);

  CHECK_CUDA( cudaFree(d_a) )
  CHECK_CUDA( cudaFree(d_b) )
  CHECK_CUDA( cudaFree(d_c) )
  CHECK_CUDA( cudaFree(d_i_idx) )
  CHECK_CUDA( cudaFree(d_o_idx) )
  free(h_a);
  free(h_b);
  free(h_c);
  free(h_i_idx);
  free(h_o_idx);
  free(h_r);
}

int main(int argc, char* argv[])
{
  if (argc != 6) {
    printf("Usage: %s <NA> <NC> <C> <num_ops> <repeat>\n", argv[0]);
    printf("<Number of rows in A> <Number of rows in C> ");
    printf("<Number of channels/columns> <Number of operations>\n");
    return 1;
  }
  const int NA = atoi(argv[1]);
  const int NC = atoi(argv[2]);
  const int  C = atoi(argv[3]);
  const int num_ops = atoi(argv[4]);
  const int repeat = atoi(argv[5]);

  if (num_ops > NC) {
     printf("Error: Number of operations is larger than number of rows in C\n");
    return 1;
  }

  // Two trials
  for (int i = 0; i < 2; i++) {
    printf("FP16 FMA\n");
    fma<__half>(NA, NC, C, num_ops, repeat);

    printf("BF16 FMA\n");
    fma<__nv_bfloat16>(NA, NC, C, num_ops, repeat);

    printf("FP32 FMA\n");
    fma<float>(NA, NC, C, num_ops, repeat);

    printf("FP64 FMA\n");
    fma<double>(NA, NC, C, num_ops, repeat);
  }

  if (!ok) return 1;
  return 0;
}
