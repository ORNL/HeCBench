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
#include <stdio.h>  // fopen
#include <stdlib.h> // EXIT_FAILURE
#include <string.h> // strtok
#include <algorithm>
#include <numeric>
#include <vector>
#include <chrono>
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include "utils.h"

#if defined(NDEBUG)
#   define PRINT_INFO(var)
#else
#   define PRINT_INFO(var) printf("  " #var ": %f\n", var);
#endif

namespace mkl_sparse = oneapi::mkl::sparse;

// Sort the column indices (and the matching values) of every row in ascending
// order. The mtx parser only orders the entries by row, while the ILU(0) below
// needs sorted columns to tell the L and U part of a row apart.
static void sort_csr_columns(int m, const int* rows, int* columns,
                             double* values) {
    std::vector<int> perm;
    std::vector<int> col_tmp;
    std::vector<double> val_tmp;
    for (int i = 0; i < m; i++) {
        const int begin = rows[i];
        const int n     = rows[i + 1] - begin;
        perm.resize(n);
        std::iota(perm.begin(), perm.end(), 0);
        std::sort(perm.begin(), perm.end(),
                  [&](int a, int b) {
                      return columns[begin + a] < columns[begin + b];
                  });
        col_tmp.assign(columns + begin, columns + begin + n);
        val_tmp.assign(values + begin, values + begin + n);
        for (int k = 0; k < n; k++) {
            columns[begin + k] = col_tmp[perm[k]];
            values[begin + k]  = val_tmp[perm[k]];
        }
    }
}

// Incomplete-LU factorization with zero fill-in, the equivalent of
// cusparseDcsrilu02. oneMKL has no sparse ILU routine for SYCL devices, so the
// preconditioner is built on the host. The factors overwrite 'values' in place:
// the strictly lower part holds L (unit diagonal) and the remainder holds U.
// Returns the row index of the first structural/numerical zero pivot, or -1.
static int csrilu0(int m, const int* rows, const int* columns, double* values) {
    std::vector<int> diag(m, -1); // position of the diagonal entry of each row
    std::vector<int> pos(m, -1);  // position of column j in the current row
    for (int i = 0; i < m; i++) {
        for (int k = rows[i]; k < rows[i + 1]; k++)
            pos[columns[k]] = k;
        for (int k = rows[i]; k < rows[i + 1]; k++) {
            const int j = columns[k];
            if (j >= i)
                break;
            if (diag[j] == -1)
                return j;
            const double factor = values[k] / values[diag[j]];
            values[k] = factor;
            // subtract factor * U(j, j+1:m) from the current row
            for (int l = diag[j] + 1; l < rows[j + 1]; l++) {
                const int p = pos[columns[l]];
                if (p != -1)
                    values[p] -= factor * values[l];
            }
        }
        diag[i] = pos[i];
        if (diag[i] == -1 || values[diag[i]] == 0.0)
            return i;
        for (int k = rows[i]; k < rows[i + 1]; k++)
            pos[columns[k]] = -1;
    }
    return -1;
}

double gpu_BiCGStab(
                 sycl::queue&              q,
                 int                       verbose,
                 int                       m,
                 mkl_sparse::matrix_handle_t matA,
                 mkl_sparse::matrix_handle_t matM_lower,
                 mkl_sparse::matrix_handle_t matM_upper,
                 double*                   d_B,
                 double*                   d_X,
                 double*                   d_R0,
                 double*                   d_R,
                 double*                   d_P,
                 double*                   d_P_aux,
                 double*                   d_S,
                 double*                   d_S_aux,
                 double*                   d_V,
                 double*                   d_T,
                 double*                   d_tmp,
                 double*                   d_result,
                 int                       maxIterations,
                 double                    tolerance) {
    const double zero      = 0.0;
    const double one       = 1.0;
    const double minus_one = -1.0;
    const auto nontrans    = oneapi::mkl::transpose::nontrans;
    const auto lower       = oneapi::mkl::uplo::lower;
    const auto upper       = oneapi::mkl::uplo::upper;
    const auto unit        = oneapi::mkl::diag::unit;
    const auto nonunit     = oneapi::mkl::diag::nonunit;

    // A wait is required before the host reads a reduction
    // result back from the shared 'd_result' buffer.
    //--------------------------------------------------------------------------
    // Analyze the sparsity pattern of the triangular factors. The optimized
    // data is cached inside the matrix handles, hence M_lower and M_upper must
    // use two distinct handles even though they share the same CSR arrays.
    mkl_sparse::optimize_trsv(q, lower, nontrans, unit, matM_lower);
    mkl_sparse::optimize_trsv(q, upper, nontrans, nonunit, matM_upper);
    //--------------------------------------------------------------------------
    // ### 1 ### R0 = b - A * X0 (using initial guess in X)
    //    (a) copy b in R0
    q.memcpy(d_R0, d_B, m * sizeof(double));
    //    (b) compute R = -A * X0 + R
    mkl_sparse::gemv(q, nontrans, minus_one, matA, d_X, one, d_R0);
    //--------------------------------------------------------------------------
    double alpha, delta, delta_prev, omega;
    oneapi::mkl::blas::dot(q, m, d_R0, 1, d_R0, 1, d_result);
    q.wait();
    delta = *d_result;
    delta_prev = delta;
    // R = R0
    q.memcpy(d_R, d_R0, m * sizeof(double));
    //--------------------------------------------------------------------------
    // nrm_R0 = ||R||
    double nrm_R;
    oneapi::mkl::blas::nrm2(q, m, d_R0, 1, d_result);
    q.wait();
    nrm_R = *d_result;
    double threshold = tolerance * nrm_R;
    if (verbose) printf("  Initial Residual: Norm %e' threshold %e\n", nrm_R, threshold);
    //--------------------------------------------------------------------------
    // ### 2 ### repeat until convergence based on max iterations and
    //           and relative residual

    for (int i = 1; i <= maxIterations; i++) {
        if (verbose) printf("  Iteration = %d; Error Norm = %e\n", i, nrm_R);
        //----------------------------------------------------------------------
        // ### 4, 7 ### P_i = R_i
        q.memcpy(d_P, d_R, m * sizeof(double));
        if (i > 1) {
            //------------------------------------------------------------------
            // ### 6 ### beta = (delta_i / delta_i-1) * (alpha / omega_i-1)
            //    (a) delta_i = (R'_0, R_i-1)
            oneapi::mkl::blas::dot(q, m, d_R0, 1, d_R, 1, d_result);
            q.wait();
            delta = *d_result;
            //    (b) beta = (delta_i / delta_i-1) * (alpha / omega_i-1);
            double beta = (delta / delta_prev) * (alpha / omega);
            delta_prev  = delta;
            //------------------------------------------------------------------
            // ### 7 ### P = R + beta * (P - omega * V)
            //    (a) P = - omega * V + P
            double minus_omega = -omega;
            oneapi::mkl::blas::axpy(q, m, minus_omega, d_V, 1, d_P, 1);
            //    (b) P = beta * P
            oneapi::mkl::blas::scal(q, m, beta, d_P, 1);
            //    (c) P = R + P
            oneapi::mkl::blas::axpy(q, m, one, d_R, 1, d_P, 1);
        }
        //----------------------------------------------------------------------
        // ### 9 ### P_aux = M_U^-1 M_L^-1 P_i
        //    (a) M_L^-1 P_i => tmp    (triangular solver)
        q.memset(d_tmp,   0x0, m * sizeof(double));
        q.memset(d_P_aux, 0x0, m * sizeof(double));
        mkl_sparse::trsv(q, lower, nontrans, unit, one, matM_lower, d_P, d_tmp);
        //    (b) M_U^-1 tmp => P_aux    (triangular solver)
        mkl_sparse::trsv(q, upper, nontrans, nonunit, one, matM_upper, d_tmp,
                         d_P_aux);
        //----------------------------------------------------------------------
        // ### 10 ### alpha = (R'0, R_i-1) / (R'0, A * P_aux)
        //    (a) V = A * P_aux
        mkl_sparse::gemv(q, nontrans, one, matA, d_P_aux, zero, d_V);
        //    (b) denominator = R'0 * V
        oneapi::mkl::blas::dot(q, m, d_R0, 1, d_V, 1, d_result);
        q.wait();
        double denominator = *d_result;
        alpha = delta / denominator;
        if (verbose) PRINT_INFO(delta)
        if (verbose) PRINT_INFO(alpha)
        //----------------------------------------------------------------------
        // ### 11 ###  X_i = X_i-1 + alpha * P_aux
        oneapi::mkl::blas::axpy(q, m, alpha, d_P_aux, 1, d_X, 1);
        //----------------------------------------------------------------------
        // ### 12 ###  S = R_i-1 - alpha * (A * P_aux)
        //    (a) S = R_i-1
        q.memcpy(d_S, d_R, m * sizeof(double));
        //    (b) S = -alpha * V + R_i-1
        double minus_alpha = -alpha;
        oneapi::mkl::blas::axpy(q, m, minus_alpha, d_V, 1, d_S, 1);
        //----------------------------------------------------------------------
        // ### 13 ###  check ||S|| < threshold
        oneapi::mkl::blas::nrm2(q, m, d_S, 1, d_result);
        q.wait();
        double nrm_S = *d_result;
        if (verbose) PRINT_INFO(nrm_S)
        if (nrm_S < threshold)
            break;
        //----------------------------------------------------------------------
        // ### 14 ### S_aux = M_U^-1 M_L^-1 S
        //    (a) M_L^-1 S => tmp    (triangular solver)
        q.memset(d_tmp,   0x0, m * sizeof(double));
        q.memset(d_S_aux, 0x0, m * sizeof(double));
        mkl_sparse::trsv(q, lower, nontrans, unit, one, matM_lower, d_S, d_tmp);
        //    (b) M_U^-1 tmp => S_aux    (triangular solver)
        mkl_sparse::trsv(q, upper, nontrans, nonunit, one, matM_upper, d_tmp,
                         d_S_aux);
        //----------------------------------------------------------------------
        // ### 15 ### omega = (A * S_aux, s) / (A * S_aux, A * S_aux)
        //    (a) T = A * S_aux
        mkl_sparse::gemv(q, nontrans, one, matA, d_S_aux, zero, d_T);
        //    (b) omega_num = (A * S_aux, s)
        double omega_num, omega_den;
        oneapi::mkl::blas::dot(q, m, d_T, 1, d_S, 1, d_result);
        q.wait();
        omega_num = *d_result;
        //    (c) omega_den = (A * S_aux, A * S_aux)
        oneapi::mkl::blas::dot(q, m, d_T, 1, d_T, 1, d_result);
        q.wait();
        omega_den = *d_result;
        //    (d) omega = omega_num / omega_den
        omega = omega_num / omega_den;
        if (verbose) PRINT_INFO(omega)
        // ---------------------------------------------------------------------
        // ### 16 ### omega = X_i = X_i-1 + alpha * P_aux + omega * S_aux
        //    (a) X_i has been updated with h = X_i-1 + alpha * P_aux
        //        X_i = omega * S_aux + X_i
        oneapi::mkl::blas::axpy(q, m, omega, d_S_aux, 1, d_X, 1);
        // ---------------------------------------------------------------------
        // ### 17 ###  R_i+1 = S - omega * (A * S_aux)
        //    (a) copy S in R
        q.memcpy(d_R, d_S, m * sizeof(double));
        //    (a) R_i+1 = -omega * T + R
        double minus_omega = -omega;
        oneapi::mkl::blas::axpy(q, m, minus_omega, d_T, 1, d_R, 1);
       // ---------------------------------------------------------------------
        // ### 18 ###  check ||R_i|| < threshold
        oneapi::mkl::blas::nrm2(q, m, d_R, 1, d_result);
        q.wait();
        nrm_R = *d_result;
        if (verbose) PRINT_INFO(nrm_R)
        if (nrm_R < threshold)
            break;
    }
    //--------------------------------------------------------------------------
    if (verbose) printf("Check Solution\n"); // ||R = b - A * X||
    //    (a) copy b in R
    q.memcpy(d_R, d_B, m * sizeof(double));
    // R = -A * X + R
    mkl_sparse::gemv(q, nontrans, minus_one, matA, d_X, one, d_R);
    // check ||R||
    oneapi::mkl::blas::nrm2(q, m, d_R, 1, d_result);
    q.wait();
    nrm_R = *d_result;
    //--------------------------------------------------------------------------
    return nrm_R;
}

//==============================================================================
//==============================================================================

int main(int argc, char** argv) {
    const double tolerance     = 0.0000000001;
    if (argc != 4) {
        printf("Usage: %s <matrix.mtx> <maximum number of iterations> <verbose output>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char *file_path = argv[1];
    const int maxIterations = atoi(argv[2]);
    const int verbose = atoi(argv[3]);

    int base = 0;
    int num_rows, num_cols, nnz, num_lines, is_symmetric;
    mtx_header(file_path, &num_lines, &num_rows, &num_cols, &nnz, &is_symmetric);
    printf("\nmatrix name: %s\n"
           "num. rows:   %d\n"
           "num. cols:   %d\n"
           "nnz:         %d\n"
           "structure:   %s\n\n",
           file_path, num_rows, num_cols, nnz,
           (is_symmetric) ? "symmetric" : "unsymmetric");
    if (num_rows != num_cols) {
        printf("the input matrix must be square\n");
        return EXIT_FAILURE;
    }
    if (!is_symmetric) {
        printf("the input matrix must be symmetric\n");
        return EXIT_FAILURE;
    }
    int     m           = num_rows;
    int     num_offsets = m + 1;
    int*    h_A_rows    = (int*)    malloc(num_offsets * sizeof(int));
    int*    h_A_columns = (int*)    malloc(nnz * sizeof(int));
    double* h_A_values  = (double*) malloc(nnz * sizeof(double));
    double* h_M_values  = (double*) malloc(nnz * sizeof(double));
    double* h_X         = (double*) malloc(m * sizeof(double));
    printf("Matrix parsing...\n");
    nnz = mtx_parsing(file_path, num_lines, num_rows, nnz, h_A_rows,
                      h_A_columns, h_A_values, base);
    sort_csr_columns(m, h_A_rows, h_A_columns, h_A_values);
    printf("Testing BiCGStab\n");
    for (int i = 0; i < num_rows; i++)
        h_X[i] = 1.0;
    //--------------------------------------------------------------------------
    // Perform Incomplete-LU factorization of A (csrilu0) -> M_lower, M_upper
    // oneMKL provides no sparse ILU for SYCL devices, so the preconditioner is
    // computed on the host before it is uploaded.
    memcpy(h_M_values, h_A_values, nnz * sizeof(double));
    int zero_pivot = csrilu0(m, h_A_rows, h_A_columns, h_M_values);
    if (zero_pivot != -1)
        printf("Warning: zero pivot found at row %d\n", zero_pivot);
    //--------------------------------------------------------------------------
    // ### Device memory management ###
#ifdef USE_GPU
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
    sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

    int*    d_A_rows, *d_A_columns;
    double* d_A_values, *d_M_values;
    double *d_B, *d_X, *d_R, *d_R0, *d_P, *d_P_aux, *d_S, *d_S_aux, *d_V, *d_T,
           *d_tmp;

    // allocate device memory for CSR matrices
    d_A_rows    = sycl::malloc_device<int>(num_offsets, q);
    d_A_columns = sycl::malloc_device<int>(nnz, q);
    d_A_values  = sycl::malloc_device<double>(nnz, q);
    d_M_values  = sycl::malloc_device<double>(nnz, q);

    d_B     = sycl::malloc_device<double>(m, q);
    d_X     = sycl::malloc_device<double>(m, q);
    d_R     = sycl::malloc_device<double>(m, q);
    d_R0    = sycl::malloc_device<double>(m, q);
    d_P     = sycl::malloc_device<double>(m, q);
    d_P_aux = sycl::malloc_device<double>(m, q);
    d_S     = sycl::malloc_device<double>(m, q);
    d_S_aux = sycl::malloc_device<double>(m, q);
    d_V     = sycl::malloc_device<double>(m, q);
    d_T     = sycl::malloc_device<double>(m, q);
    d_tmp   = sycl::malloc_device<double>(m, q);

    // scalar results of the BLAS reductions are read back on the host
    double* d_result = sycl::malloc_shared<double>(1, q);

    // copy the CSR matrices and vectors into device memory
    q.memcpy(d_A_rows, h_A_rows, num_offsets * sizeof(int));
    q.memcpy(d_A_columns, h_A_columns, nnz * sizeof(int));
    q.memcpy(d_A_values, h_A_values, nnz * sizeof(double));
    q.memcpy(d_M_values, h_M_values, nnz * sizeof(double));
    q.memcpy(d_X, h_X, m * sizeof(double));
    //--------------------------------------------------------------------------
    // ### oneMKL sparse matrix handles initialization ###
    // IMPORTANT: Upper/Lower triangular decompositions of A
    //            (matM_lower, matM_upper) must use two distinct handles
    mkl_sparse::matrix_handle_t matA       = nullptr;
    mkl_sparse::matrix_handle_t matM_lower = nullptr;
    mkl_sparse::matrix_handle_t matM_upper = nullptr;
    mkl_sparse::init_matrix_handle(&matA);
    mkl_sparse::init_matrix_handle(&matM_lower);
    mkl_sparse::init_matrix_handle(&matM_upper);

    const auto baseIdx  = oneapi::mkl::index_base::zero;
    const auto nontrans = oneapi::mkl::transpose::nontrans;
    int* d_M_rows       = d_A_rows;
    int* d_M_columns    = d_A_columns;
    // A
    mkl_sparse::set_csr_data(q, matA, m, m, nnz, baseIdx, d_A_rows, d_A_columns,
                             d_A_values);
    // M_lower and M_upper share the ILU(0) factors; the fill mode and the
    // diagonal type are arguments of the triangular solver in oneMKL
    mkl_sparse::set_csr_data(q, matM_lower, m, m, nnz, baseIdx, d_M_rows,
                             d_M_columns, d_M_values);
    mkl_sparse::set_csr_data(q, matM_upper, m, m, nnz, baseIdx, d_M_rows,
                             d_M_columns, d_M_values);

    mkl_sparse::set_matrix_property(matA, mkl_sparse::property::sorted);
    mkl_sparse::set_matrix_property(matM_lower, mkl_sparse::property::sorted);
    mkl_sparse::set_matrix_property(matM_upper, mkl_sparse::property::sorted);
    //--------------------------------------------------------------------------
    // ### Preparation ### b = A * X
    const double alpha = 0.75;
    double beta = 0.0;
    mkl_sparse::optimize_gemv(q, nontrans, matA);

    mkl_sparse::gemv(q, nontrans, alpha, matA, d_X, beta, d_B);
    // X0 = 0
    q.memset(d_X, 0x0, m * sizeof(double));
    //--------------------------------------------------------------------------
    // ### Run BiCGStab computation ###
    printf("BiCGStab loop:\n");

    auto start = std::chrono::steady_clock::now();

    double nrm_R = gpu_BiCGStab(q, verbose, m, matA, matM_lower, matM_upper,
                                d_B, d_X, d_R0, d_R, d_P, d_P_aux, d_S, d_S_aux,
                                d_V, d_T, d_tmp, d_result, maxIterations,
                                tolerance);

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Total execution time of BiCGStab: %f (s)\n", time * 1e-9f);
    printf("Final error norm = %e\n", nrm_R);
    //--------------------------------------------------------------------------
    // ### Free resources ###
    mkl_sparse::release_matrix_handle(q, &matA);
    mkl_sparse::release_matrix_handle(q, &matM_lower);
    mkl_sparse::release_matrix_handle(q, &matM_upper);
    q.wait();

    free(h_A_rows);
    free(h_A_columns);
    free(h_A_values);
    free(h_M_values);
    free(h_X);

    sycl::free(d_X, q);
    sycl::free(d_B, q);
    sycl::free(d_R, q);
    sycl::free(d_R0, q);
    sycl::free(d_P, q);
    sycl::free(d_P_aux, q);
    sycl::free(d_S, q);
    sycl::free(d_S_aux, q);
    sycl::free(d_V, q);
    sycl::free(d_T, q);
    sycl::free(d_tmp, q);
    sycl::free(d_A_values, q);
    sycl::free(d_A_columns, q);
    sycl::free(d_A_rows, q);
    sycl::free(d_M_values, q);
    sycl::free(d_result, q);
    return EXIT_SUCCESS;
}
