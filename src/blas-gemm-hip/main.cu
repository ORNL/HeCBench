#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <type_traits>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hipblas/hipblas.h>
#include "utils.h"

#define TILE_X 16
#define TILE_Y 16

// M * K, K * N
template <typename T>
__global__ void matrix_mul(T *a, T *b, T *c, int M, int K, int N, T alpha, T beta) {
  int row = blockIdx.y * TILE_Y + threadIdx.y;
  int col = blockIdx.x * TILE_X + threadIdx.x;
  if (row < M && col < N) {
    T s = 0;
    for (int k = 0; k < K; k++)
      s += a[row * K + k] * b[k * N + col];
    c[row * N + col] = alpha * s + beta * c[row * N + col];
  }
}

template <typename T>
void run_simple_gemm(T *a, T *b, T *c, int M, int K, int N, T alpha, T beta) {
  dim3 grids ((N + TILE_X - 1) / TILE_X, (M + TILE_Y - 1) / TILE_Y);
  dim3 blocks (TILE_X, TILE_Y);
  matrix_mul<<<grids, blocks>>>(a, b, c, M, K, N, alpha, beta);
}


//
// Main example for Gemm consisting of
// initialization of A, B and C matrices as well as
// scalars alpha and beta.  Then the product
//
// C = alpha * op(A) * op(B) + beta * C
//
// is performed and finally the results are post processed.
//
template <typename fp>
void run_gemm_example(int m, int k, int n, int repeat) {

  //
  // Initialize data for Gemm
  //
  // C = alpha * op(A) * op(B)  + beta * C
  //

  // set scalar fp values
  const fp alpha = fp(2.0);
  const fp beta  = fp(0.5);

  const size_t A_size = sizeof(fp) * m * k;
  const size_t B_size = sizeof(fp) * k * n;
  const size_t C_size = sizeof(fp) * m * n;

  // prepare matrix data
  fp* a = (fp *) aligned_alloc(64, A_size);
  fp* b = (fp *) aligned_alloc(64, B_size);
  fp* c = (fp *) aligned_alloc(64, C_size);
  fp* r = (fp *) aligned_alloc(64, C_size);

  srand(2);
  rand_matrix(a, m, k);
  rand_matrix(b, k, n);
  rand_matrix(c, m, n);

  fp *da, *db, *dc, *dr;
  hipMalloc((void**)&da, A_size);
  hipMalloc((void**)&db, B_size);
  hipMalloc((void**)&dc, C_size);
  hipMalloc((void**)&dr, C_size);
  hipMemcpy(da, a, A_size, hipMemcpyHostToDevice);
  hipMemcpy(db, b, B_size, hipMemcpyHostToDevice);
  hipMemcpy(dc, c, C_size, hipMemcpyHostToDevice);
  hipMemcpy(dr, c, C_size, hipMemcpyHostToDevice);

  // create execution queue and buffers of matrix data
  hipblasHandle_t h;
  hipblasCreate(&h);

  std::cout << "Checking BLAS GEMM.. ";
  run_simple_gemm(da, db, dr, m, k, n, alpha, beta);

  if constexpr (std::is_same_v<fp, __half>)
    hipblasHgemm(h, HIPBLAS_OP_N, HIPBLAS_OP_N, n, m, k,
                &alpha, db, n, da, k, &beta, dc, n);
  else if constexpr (std::is_same_v<fp, float>)
    hipblasSgemm(h, HIPBLAS_OP_N, HIPBLAS_OP_N, n, m, k,
                &alpha, db, n, da, k, &beta, dc, n);
  else if constexpr (std::is_same_v<fp, double>)
    hipblasDgemm(h, HIPBLAS_OP_N, HIPBLAS_OP_N, n, m, k,
                &alpha, db, n, da, k, &beta, dc, n);

  hipMemcpy(c, dc, C_size, hipMemcpyDeviceToHost);
  hipMemcpy(r, dr, C_size, hipMemcpyDeviceToHost);
  int error = memcmp(c, r, C_size);
  std::cout << (error ? "FAIL" : "PASS") << std::endl;
  if (error) exit(1);

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    if constexpr (std::is_same_v<fp, __half>)
      hipblasHgemm(h, HIPBLAS_OP_N, HIPBLAS_OP_N, n, m, k,
                  &alpha, db, n, da, k, &beta, dc, n);
    else if constexpr (std::is_same_v<fp, float>)
      hipblasSgemm(h, HIPBLAS_OP_N, HIPBLAS_OP_N, n, m, k,
                  &alpha, db, n, da, k, &beta, dc, n);
    else if constexpr (std::is_same_v<fp, double>)
      hipblasDgemm(h, HIPBLAS_OP_N, HIPBLAS_OP_N, n, m, k,
                  &alpha, db, n, da, k, &beta, dc, n);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  double time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  performance(m, n, k, false, time / repeat);

  //
  // Post Processing
  //

#ifdef DEBUG
  std::cout << "\n\t\tOutputting 2x2 block of A,B,C matrices:" << std::endl;

  // output the top 2x2 block of A matrix
  print_2x2_matrix_values(a, k, "A");

  // output the top 2x2 block of B matrix
  print_2x2_matrix_values(b, n, "B");

  // output the top 2x2 block of C matrix
  hipMemcpy(c, dc, C_size, hipMemcpyDeviceToHost);
  print_2x2_matrix_values(c, n, "C");
#endif

  hipblasDestroy(h);

  hipFree(da);
  hipFree(db);
  hipFree(dc);
  hipFree(dr);

  free(a);
  free(b);
  free(c);
  free(r);
}

//
// Main entry point for example.
//
int main (int argc, char ** argv) {
  if (argc != 5) {
    printf("Usage: %s <m> <k> <n> <repeat>\n", argv[0]);
    return 1;
  }
  const int m = atoi(argv[1]);
  const int k = atoi(argv[2]);
  const int n = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  std::cout << "\tRunning with half precision data type:" << std::endl;
  run_gemm_example<__half>(m, k, n, repeat);

  std::cout << "\tRunning with single precision data type:" << std::endl;
  run_gemm_example<float>(m, k, n, repeat);

  std::cout << "\tRunning with double precision data type:" << std::endl;
  run_gemm_example<double>(m, k, n, repeat);

  return 0;
}
