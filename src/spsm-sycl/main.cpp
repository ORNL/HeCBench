#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <math.h>
#include <chrono>
#include <oneapi/mkl.hpp>     // oneapi::mkl::sparse::trsm
#include "utils.h"

int main(int argc, char *argv[])
{
  int repeat = 1;

  if (argc != 6) {
    printf("The function solves a system of linear equations whose ");
    printf("coefficients are represented in a sparse triangular matrix.\n");
    printf("The sparse matrix is represented in CSR (Compressed Sparse Row) storage format\n");

    printf("Usage %s <M> <N> <A_nnz> <repeat> <verify>\n", argv[0]);
    printf("SPSM (A, B, C) where (A: M * M, C: M * N, B: M * N)\n");
    return 1;
  }

  int m, n, a_nnz, verify;

  m = atoi(argv[1]);
  n = atoi(argv[2]);
  a_nnz = atoi(argv[3]);
  repeat = atoi(argv[4]);
  verify = atoi(argv[5]);

  // Host problem definition
  const int A_num_rows = m;  // a square matrix
  const int A_num_cols = m;
  int       A_nnz      = a_nnz;
  const int lda        = A_num_cols;
  const int A_size     = lda * A_num_rows;

  const int C_num_rows = m;
  const int C_num_cols = n;
  const int ldc        = C_num_cols;
  const int C_size     = ldc * C_num_rows;

  const int B_num_rows = m;
  const int B_num_cols = n;
  const int ldb        = B_num_cols;
  const int B_size     = ldb * B_num_rows;

  const int nrhs = n;

  float *hA = (float*) malloc (A_size * sizeof(float));
  float *hB = (float*) malloc (B_size * sizeof(float));
  float *hC = (float*) malloc (C_size * sizeof(float));

  const size_t A_value_size_bytes  = A_nnz * sizeof(float);
  const size_t A_colidx_size_bytes = A_nnz * sizeof(int);
  const size_t A_rowidx_size_bytes = (A_num_rows + 1) * sizeof(int);

  float *hA_values = (float*) malloc (A_value_size_bytes);
  int *hA_columns = (int*) malloc (A_colidx_size_bytes);
  int *hA_offsets = (int*) malloc (A_rowidx_size_bytes);

  printf("Initializing host matrices..\n");
  A_nnz = init_lower_triangular_matrix(hA, A_num_rows, A_nnz);
  init_csr(hA_offsets, hA_values, hA_columns, hA,
           A_num_rows, A_num_cols, A_nnz);

  init_matrix(hC, C_num_rows, C_num_cols, C_size);

  // precompute hB
  spsm (hA, hC, hB, A_num_rows, C_num_cols);
  printf("Done\n");

  float alpha = 1.0f;

  //--------------------------------------------------------------------------
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // Device memory management
  int   *dA_offsets, *dA_columns;
  float *dA_values, *dB, *dC;

  dA_offsets = sycl::malloc_device<int>(A_num_rows + 1, q);
  dA_columns = sycl::malloc_device<int>(A_nnz, q);
  dA_values  = sycl::malloc_device<float>(A_nnz, q);
  dB = sycl::malloc_device<float>(nrhs * A_num_cols, q);
  dC = sycl::malloc_device<float>(nrhs * A_num_rows, q);

  q.memcpy(dA_offsets, hA_offsets, A_rowidx_size_bytes);
  q.memcpy(dA_columns, hA_columns, A_colidx_size_bytes);
  q.memcpy(dA_values,  hA_values,  A_value_size_bytes);
  q.memcpy(dB, hB, nrhs * A_num_cols * sizeof(float));
  q.memcpy(dC, hC, nrhs * A_num_rows * sizeof(float));
  q.wait();
  //--------------------------------------------------------------------------
  // oneMKL sparse APIs

  // Create sparse matrix A in CSR format
  oneapi::mkl::sparse::matrix_handle_t matA = nullptr;
  oneapi::mkl::sparse::init_matrix_handle(&matA);
  oneapi::mkl::sparse::set_csr_data(q, matA, A_num_rows, A_num_cols, A_nnz,
                                    oneapi::mkl::index_base::zero,
                                    dA_offsets, dA_columns, dA_values);

  // Dense matrices B and C are stored in row-major order
  const oneapi::mkl::layout layout      = oneapi::mkl::layout::row_major;
  const oneapi::mkl::transpose opA      = oneapi::mkl::transpose::nontrans;
  const oneapi::mkl::transpose opX      = oneapi::mkl::transpose::nontrans;
  // Lower fill mode with a non-unit diagonal
  const oneapi::mkl::uplo fillmode      = oneapi::mkl::uplo::lower;
  const oneapi::mkl::diag diagtype      = oneapi::mkl::diag::nonunit;

  // Analysis phase: solves op(A) * Y = alpha * op(X) with 'nrhs' columns
  oneapi::mkl::sparse::optimize_trsm(q, layout, fillmode, opA, diagtype,
                                     matA, nrhs);
  q.wait();

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    // execute SpSM : solves matA * dC = alpha * dB
    oneapi::mkl::sparse::trsm(q, layout, opA, opX, fillmode, diagtype, alpha,
                              matA, dB, nrhs, ldb, dC, ldc);
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of SpSM solve: %f (us)\n", (time * 1e-3f) / repeat);

  oneapi::mkl::sparse::release_matrix_handle(q, &matA, {});
  q.wait();
  //--------------------------------------------------------------------------
  // device result check
  if (verify) {
    printf("Checking results..\n");
    float *hY = (float*) malloc (C_size * sizeof(float));
    q.memcpy(hY, dC, nrhs * A_num_rows * sizeof(float)).wait();

    // compute hB using the device result
    float *hB2 = (float*) malloc (B_size * sizeof(float));
    spsm (hA, hY, hB2, A_num_rows, C_num_cols);

    int correct = 1;
    for (int i = 0; i < A_num_rows * C_num_cols; i++) {
      if (fabsf(hB[i] - hB2[i]) > 1e-2f) {
        printf("@%d %f != %f\n", i, hB[i], hB2[i]);
        correct = 0;
        break;
      }
    }
    if (correct)
        printf("spsm_csr_example test PASSED\n");
    else
        printf("spsm_csr_example test FAILED: wrong result\n");

    free(hB2);
    free(hY);
  }

  //--------------------------------------------------------------------------
  // device memory deallocation
  sycl::free(dA_offsets, q);
  sycl::free(dA_columns, q);
  sycl::free(dA_values, q);
  sycl::free(dB, q);
  sycl::free(dC, q);
  free(hA);
  free(hB);
  free(hC);
  free(hA_values);
  free(hA_columns);
  free(hA_offsets);
  return EXIT_SUCCESS;
}
