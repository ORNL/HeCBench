/*
 * Copyright 1993-2022 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */
#include <stdio.h>            // printf
#include <stdlib.h>           // EXIT_FAILURE
#include <math.h> 
#include <chrono>
#include <cuda_runtime_api.h> // cudaMalloc, cudaMemcpy, etc.
#include <cusparse.h>         // cusparseSpSM
#include "utils.h"

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

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

  //
  float alpha = 1.0f;

  //--------------------------------------------------------------------------
  // Device memory management
  int   *dA_offsets, *dA_columns;
  float *dA_values, *dB, *dC;
  CHECK_CUDA( cudaMalloc((void**) &dA_offsets,
                         (A_num_rows + 1) * sizeof(int)) )
  CHECK_CUDA( cudaMalloc((void**) &dA_columns, A_nnz * sizeof(int))       )
  CHECK_CUDA( cudaMalloc((void**) &dA_values,  A_nnz * sizeof(float))     )
  CHECK_CUDA( cudaMalloc((void**) &dB, nrhs * A_num_cols * sizeof(float)) )
  CHECK_CUDA( cudaMalloc((void**) &dC, nrhs * A_num_rows * sizeof(float)) )

  CHECK_CUDA( cudaMemcpy(dA_offsets, hA_offsets,
                         (A_num_rows + 1) * sizeof(int),
                         cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                         cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(dA_values, hA_values, A_nnz * sizeof(float),
                         cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(dB, hB, nrhs * A_num_cols * sizeof(float),
                         cudaMemcpyHostToDevice) )
  CHECK_CUDA( cudaMemcpy(dC, hC, nrhs * A_num_rows * sizeof(float),
                         cudaMemcpyHostToDevice) )
  //--------------------------------------------------------------------------
  // CUSPARSE APIs
  cusparseHandle_t     handle = NULL;
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;
  void*                dBuffer    = NULL;
  size_t               bufferSize = 0;
  cusparseSpSMDescr_t  spsmDescr;
  CHECK_CUSPARSE( cusparseCreate(&handle) )
  // Create sparse matrix A in CSR format
  CHECK_CUSPARSE( cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                    dA_offsets, dA_columns, dA_values,
                                    CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                    CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F) )
  // Create dense vector B'
  CHECK_CUSPARSE( cusparseCreateDnMat(&matB, A_num_cols, nrhs, ldb, dB,
                                      CUDA_R_32F, CUSPARSE_ORDER_ROW) )
  // Create dense vector C'
  CHECK_CUSPARSE( cusparseCreateDnMat(&matC, A_num_rows, nrhs, ldc, dC,
                                      CUDA_R_32F, CUSPARSE_ORDER_ROW) )
  // Create opaque data structure, that holds analysis data between calls.
  CHECK_CUSPARSE( cusparseSpSM_createDescr(&spsmDescr) )
  // Specify Lower|Upper fill mode.
  cusparseFillMode_t fillmode = CUSPARSE_FILL_MODE_LOWER;
  CHECK_CUSPARSE( cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_FILL_MODE,
                                            &fillmode, sizeof(fillmode)) )
  // Specify Unit|Non-Unit diagonal type.
  cusparseDiagType_t diagtype = CUSPARSE_DIAG_TYPE_NON_UNIT;
  CHECK_CUSPARSE( cusparseSpMatSetAttribute(matA, CUSPARSE_SPMAT_DIAG_TYPE,
                                            &diagtype, sizeof(diagtype)) )
  // allocate an external buffer for analysis
  CHECK_CUSPARSE( cusparseSpSM_bufferSize(
                              handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                              &alpha, matA, matB, matC, CUDA_R_32F,
                              CUSPARSE_SPSM_ALG_DEFAULT, spsmDescr,
                              &bufferSize) )
  CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) )
  CHECK_CUSPARSE( cusparseSpSM_analysis(
                              handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                              &alpha, matA, matB, matC, CUDA_R_32F,
                              CUSPARSE_SPSM_ALG_DEFAULT, spsmDescr, dBuffer) )

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    // execute SpSM
    CHECK_CUSPARSE( cusparseSpSM_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       CUSPARSE_OPERATION_NON_TRANSPOSE,
                                       &alpha, matA, matB, matC, CUDA_R_32F,
                                       CUSPARSE_SPSM_ALG_DEFAULT, spsmDescr) )
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of SpSM solve: %f (us)\n", (time * 1e-3f) / repeat);

  // destroy matrix/vector descriptors
  CHECK_CUSPARSE( cusparseDestroySpMat(matA) )
  CHECK_CUSPARSE( cusparseDestroyDnMat(matB) )
  CHECK_CUSPARSE( cusparseDestroyDnMat(matC) )
  CHECK_CUSPARSE( cusparseSpSM_destroyDescr(spsmDescr));
  CHECK_CUSPARSE( cusparseDestroy(handle) )
  //--------------------------------------------------------------------------
  // device result check
  if (verify) {
    printf("Checking results..\n");
    float *hY = (float*) malloc (C_size * sizeof(float));
    CHECK_CUDA( cudaMemcpy(hY, dC, nrhs * A_num_rows * sizeof(float),
                           cudaMemcpyDeviceToHost) )

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
  CHECK_CUDA( cudaFree(dBuffer) )
  CHECK_CUDA( cudaFree(dA_offsets) )
  CHECK_CUDA( cudaFree(dA_columns) )
  CHECK_CUDA( cudaFree(dA_values) )
  CHECK_CUDA( cudaFree(dB) )
  CHECK_CUDA( cudaFree(dC) )
  free(hA);
  free(hB);
  free(hC);
  free(hA_values);
  free(hA_columns);
  free(hA_offsets);
  return EXIT_SUCCESS;
}
