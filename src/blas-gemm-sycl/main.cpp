/*******************************************************************************
* Copyright 2018-2020 Intel Corporation.
*
* This software and the related documents are Intel copyrighted  materials,  and
* your use of  them is  governed by the  express license  under which  they were
* provided to you (License).  Unless the License provides otherwise, you may not
* use, modify, copy, publish, distribute,  disclose or transmit this software or
* the related documents without Intel's prior written permission.
*
* This software and the related documents  are provided as  is,  with no express
* or implied  warranties,  other  than those  that are  expressly stated  in the
* License.
*******************************************************************************/

/*
*
*  Content:
*       This example demonstrates use of DPCPP API mkl::blas::gemm to perform General
*       Matrix-Matrix Multiplication on a SYCL device (HOST, CPU, GPU).
*
*       C = alpha * op(A) * op(B) + beta * C
*
*       where op() is defined by one of mkl::transpose::{nontrans,trans,conjtrans}
*
*
*       The supported fping point data types for gemm matrix data are:
*           half
*           fp
*           double
*           std::complex<fp>
*           std::complex<double>
*
*
*******************************************************************************/

// stl includes
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>

// mkl/sycl includes
#include <sycl/sycl.hpp>
#include "oneapi/mkl/blas.hpp"
#include "mkl.h"

#include "utils.h"

#define TILE_X 16
#define TILE_Y 16

// M * K, K * N
template <typename T>
void matrix_mul(sycl::nd_item<2> &item, T *a, T *b, T *c, int M, int K, int N, T alpha, T beta) {
  int row = item.get_global_id(0);
  int col = item.get_global_id(1);
  if (row < M && col < N) {
    T s = 0;
    for (int k = 0; k < K; k++)
      s += a[row * K + k] * b[k * N + col];
    c[row * N + col] = alpha * s + beta * c[row * N + col];
  }
}

template <typename T>
void run_simple_gemm(sycl::queue &q, T *a, T *b, T *c, int M, int K, int N, T alpha, T beta) {
  sycl::range<2> gws ((M + TILE_Y - 1) / TILE_Y * TILE_Y,
                      (N + TILE_X - 1) / TILE_X * TILE_X);
  sycl::range<2> lws (TILE_Y, TILE_X);

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
      matrix_mul(item, a, b, c, M, K, N, alpha, beta);
    });
  });
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
void run_gemm_example(MKL_INT m, MKL_INT k, MKL_INT n, int repeat) {

  //
  // Initialize data for Gemm
  //
  // C = alpha * op(A) * op(B)  + beta * C
  //

  oneapi::mkl::transpose transA = oneapi::mkl::transpose::nontrans;
  oneapi::mkl::transpose transB = oneapi::mkl::transpose::nontrans;

  // set scalar fp values
  fp alpha = fp(2.0);
  fp beta  = fp(0.5);

  const size_t A_size = sizeof(fp) * m * k;
  const size_t B_size = sizeof(fp) * k * n;
  const size_t C_size = sizeof(fp) * m * n;

  // prepare matrix data
  fp* a = (fp *)mkl_malloc(A_size, 64);
  fp* b = (fp *)mkl_malloc(B_size, 64);
  fp* c = (fp *)mkl_malloc(C_size, 64);
  fp* r = (fp *)mkl_malloc(C_size, 64);

  srand(2);
  rand_matrix(a, m, k);
  rand_matrix(b, k, n);
  rand_matrix(c, m, n);

  // create execution queue and buffers of matrix data
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  fp *da, *db, *dc, *dr;
  da = sycl::malloc_device<fp>(m*k, q);
  db = sycl::malloc_device<fp>(k*n, q);
  dc = sycl::malloc_device<fp>(m*n, q);
  dr = sycl::malloc_device<fp>(m*n, q);
  q.memcpy(da, a, A_size);
  q.memcpy(db, b, B_size);
  q.memcpy(dc, c, B_size);
  q.memcpy(dr, c, B_size);

  std::cout << "Checking BLAS GEMM.. ";
  run_simple_gemm(q, da, db, dr, m, k, n, alpha, beta);

  oneapi::mkl::blas::gemm(q, transA, transB,
                          n, m, k, alpha, db, n, da, k, beta, dc, n);
  q.memcpy(c, dc, C_size).wait();
  q.memcpy(r, dr, C_size).wait();
  int error = memcmp(c, r, C_size);
  std::cout << (error ? "FAIL" : "PASS") << std::endl;

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    oneapi::mkl::blas::gemm(q, transA, transB, n, m, k,
                            alpha, db, n, da, k, beta, dc, n);
  }

  q.wait();
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
  q.memcpy(c, dc, C_size).wait();
  print_2x2_matrix_values(c, n, "C");
#endif

  sycl::free(da, q);
  sycl::free(db, q);
  sycl::free(dc, q);
  sycl::free(dr, q);

  mkl_free(a);
  mkl_free(b);
  mkl_free(c);
  mkl_free(r);
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
  run_gemm_example<sycl::half>(m, k, n, repeat);

  std::cout << "\tRunning with single precision data type:" << std::endl;
  run_gemm_example<float>(m, k, n, repeat);

  std::cout << "\tRunning with double precision data type:" << std::endl;
  run_gemm_example<double>(m, k, n, repeat);

  if (error) return 1;
  return 0;
}
