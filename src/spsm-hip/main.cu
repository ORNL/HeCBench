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
#include <hip/hip_runtime_api.h> // hipMalloc, hipMemcpy, etc.
#include <hipsparse/hipsparse.h>         // cusparseSpSM
#include "utils.h"

#define CHECK_HIP(func)                                                  \
{                                                                        \
    hipError_t status = (func);                                          \
    if (status != hipSuccess) {                                          \
        printf("HIP API failed at line %d with error: %s (%d)\n",        \
               __LINE__, hipGetErrorString(status), status);             \
        return EXIT_FAILURE;                                             \
    }                                                                    \
}

#define CHECK_HIPSPARSE_ERROR_CASE__(token_) \
    case token_:                             \
        fprintf(stderr, "HIPSPARSE API failed at line %d with error: %s\n",         \
               __LINE__, #token_); \
        break

#define CHECK_HIPSPARSE(error)                                                      \
    {                                                                                     \
        auto local_error = (error);                                                       \
        if(local_error != HIPSPARSE_STATUS_SUCCESS)                                       \
        {                                                                                 \
            fprintf(stderr, "hipSPARSE error: ");                                         \
            switch(local_error)                                                           \
            {                                                                             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_SUCCESS);                   \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_NOT_INITIALIZED);           \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_ALLOC_FAILED);              \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_INVALID_VALUE);             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_ARCH_MISMATCH);             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_MAPPING_ERROR);             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_EXECUTION_FAILED);          \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_INTERNAL_ERROR);            \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED); \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_ZERO_PIVOT);                \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_NOT_SUPPORTED);             \
                CHECK_HIPSPARSE_ERROR_CASE__(HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES);    \
            }                                                                             \
            fprintf(stderr, "\n");                                                        \
            return local_error;                                                           \
        }                                                                                 \
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
  CHECK_HIP( hipMalloc((void**) &dA_offsets,
                         (A_num_rows + 1) * sizeof(int)) )
  CHECK_HIP( hipMalloc((void**) &dA_columns, A_nnz * sizeof(int))       )
  CHECK_HIP( hipMalloc((void**) &dA_values,  A_nnz * sizeof(float))     )
  CHECK_HIP( hipMalloc((void**) &dB, nrhs * A_num_cols * sizeof(float)) )
  CHECK_HIP( hipMalloc((void**) &dC, nrhs * A_num_rows * sizeof(float)) )

  CHECK_HIP( hipMemcpy(dA_offsets, hA_offsets,
                         (A_num_rows + 1) * sizeof(int),
                         hipMemcpyHostToDevice) )
  CHECK_HIP( hipMemcpy(dA_columns, hA_columns, A_nnz * sizeof(int),
                         hipMemcpyHostToDevice) )
  CHECK_HIP( hipMemcpy(dA_values, hA_values, A_nnz * sizeof(float),
                         hipMemcpyHostToDevice) )
  CHECK_HIP( hipMemcpy(dB, hB, nrhs * A_num_cols * sizeof(float),
                         hipMemcpyHostToDevice) )
  CHECK_HIP( hipMemcpy(dC, hC, nrhs * A_num_rows * sizeof(float),
                         hipMemcpyHostToDevice) )
  //--------------------------------------------------------------------------
  // HIPSPARSE APIs
  hipsparseHandle_t     handle = NULL;
  hipsparseSpMatDescr_t matA;
  hipsparseDnMatDescr_t matB, matC;
  void*                 dBuffer    = NULL;
  size_t                bufferSize = 0;
  hipsparseSpSMDescr_t  spsmDescr;
  CHECK_HIPSPARSE( hipsparseCreate(&handle) )
  // Create sparse matrix A in CSR format
  CHECK_HIPSPARSE( hipsparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                      dA_offsets, dA_columns, dA_values,
                                      HIPSPARSE_INDEX_32I, HIPSPARSE_INDEX_32I,
                                      HIPSPARSE_INDEX_BASE_ZERO, HIP_R_32F) )
  // Create dense vector B'
  CHECK_HIPSPARSE( hipsparseCreateDnMat(&matB, A_num_cols, nrhs, ldb, dB,
                                        HIP_R_32F, HIPSPARSE_ORDER_ROW) )
  // Create dense vector C'
  CHECK_HIPSPARSE( hipsparseCreateDnMat(&matC, A_num_rows, nrhs, ldc, dC,
                                        HIP_R_32F, HIPSPARSE_ORDER_ROW) )
  // Create opaque data structure, that holds analysis data between calls.
  CHECK_HIPSPARSE( hipsparseSpSM_createDescr(&spsmDescr) )
  // Specify Lower|Upper fill mode.
  hipsparseFillMode_t fillmode = HIPSPARSE_FILL_MODE_LOWER;
  CHECK_HIPSPARSE( hipsparseSpMatSetAttribute(matA, HIPSPARSE_SPMAT_FILL_MODE,
                                              &fillmode, sizeof(fillmode)) )
  // Specify Unit|Non-Unit diagonal type.
  hipsparseDiagType_t diagtype = HIPSPARSE_DIAG_TYPE_NON_UNIT;
  CHECK_HIPSPARSE( hipsparseSpMatSetAttribute(matA, HIPSPARSE_SPMAT_DIAG_TYPE,
                                              &diagtype, sizeof(diagtype)) )
  // allocate an external buffer for analysis
  CHECK_HIPSPARSE( hipsparseSpSM_bufferSize(
                              handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                              HIPSPARSE_OPERATION_NON_TRANSPOSE,
                              &alpha, matA, matB, matC, HIP_R_32F,
                              HIPSPARSE_SPSM_ALG_DEFAULT, spsmDescr,
                              &bufferSize) )
  CHECK_HIP( hipMalloc(&dBuffer, bufferSize) )
  CHECK_HIPSPARSE( hipsparseSpSM_analysis(
                              handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                              HIPSPARSE_OPERATION_NON_TRANSPOSE,
                              &alpha, matA, matB, matC, HIP_R_32F,
                              HIPSPARSE_SPSM_ALG_DEFAULT, spsmDescr, dBuffer) )

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    // execute SpSM
    CHECK_HIPSPARSE( hipsparseSpSM_solve(handle, HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                       HIPSPARSE_OPERATION_NON_TRANSPOSE,
                                       &alpha, matA, matB, matC, HIP_R_32F,
                                       HIPSPARSE_SPSM_ALG_DEFAULT, spsmDescr, dBuffer) )
  }
  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of SpSM solve: %f (us)\n", (time * 1e-3f) / repeat);

  // destroy matrix/vector descriptors
  CHECK_HIPSPARSE( hipsparseDestroySpMat(matA) )
  CHECK_HIPSPARSE( hipsparseDestroyDnMat(matB) )
  CHECK_HIPSPARSE( hipsparseDestroyDnMat(matC) )
  CHECK_HIPSPARSE( hipsparseSpSM_destroyDescr(spsmDescr));
  CHECK_HIPSPARSE( hipsparseDestroy(handle) )
  //--------------------------------------------------------------------------
  // device result check
  if (verify) {
    printf("Checking results..\n");
    float *hY = (float*) malloc (C_size * sizeof(float));
    CHECK_HIP( hipMemcpy(hY, dC, nrhs * A_num_rows * sizeof(float),
                           hipMemcpyDeviceToHost) )

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
  CHECK_HIP( hipFree(dBuffer) )
  CHECK_HIP( hipFree(dA_offsets) )
  CHECK_HIP( hipFree(dA_columns) )
  CHECK_HIP( hipFree(dA_values) )
  CHECK_HIP( hipFree(dB) )
  CHECK_HIP( hipFree(dC) )
  free(hA);
  free(hB);
  free(hC);
  free(hA_values);
  free(hA_columns);
  free(hA_offsets);
  return EXIT_SUCCESS;
}
