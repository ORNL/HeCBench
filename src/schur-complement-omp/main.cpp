#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <omp.h>
#include "reference.h"

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <rows> <nnz per row> <repeat>\n", argv[0]);
    return 1;
  }

  const int m = atoi(argv[1]);
  const int nnz_row = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const int nx = 8 * nnz_row + 1024;
  const double alpha = -1.0;

  std::vector<int> h_rs1, h_jc1, h_rs2, h_jc2;
  std::vector<double> h_v1, h_v2, h_D;
  gen_csr(m, nx, nnz_row, 123, h_rs1, h_jc1, h_v1);
  gen_csr(m, nx, nnz_row, 456, h_rs2, h_jc2, h_v2);
  gen_diag(nx, 789, h_D);

  const int nnz = (int)h_v1.size();
  const size_t w_elems = (size_t)m * m;

  const int m_W = m;
  const int row_dest_start = 0, col_dest_start = 0;

  int* rs1 = h_rs1.data(); int* jc1 = h_jc1.data(); double* v1 = h_v1.data();
  int* rs2 = h_rs2.data(); int* jc2 = h_jc2.data(); double* v2 = h_v2.data();
  double* D = h_D.data();
  std::vector<double> h_W(w_elems, 0.0);
  double* W = h_W.data();
  std::vector<double> h_ref(w_elems);
  int errors = 0;

  #pragma omp target enter data map(to: rs1[0:m+1], jc1[0:nnz], v1[0:nnz], \
                                        rs2[0:m+1], jc2[0:nnz], v2[0:nnz], \
                                        D[0:nx], W[0:w_elems])

  // --- diagonal block: W += alpha * J D^{-1} J^T ---------------------------

  // host/device correctness check (run once, verify against reference) before timing
  #pragma omp target teams distribute parallel for thread_limit(128)
  for (size_t k = 0; k < w_elems; k++) W[k] = 0.0;

  #pragma omp target teams distribute parallel for collapse(2) thread_limit(128)
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      if (j < i) continue;
      double acc = 0.0;
      if (i == j) {
        for (int k = rs1[i]; k < rs1[i + 1]; k++)
          acc += v1[k] / D[jc1[k]] * v1[k];
      } else {
        int ki = rs1[i], kj = rs1[j];
        while (ki < rs1[i + 1] && kj < rs1[j + 1]) {
          if (jc1[ki] == jc1[kj]) { acc += v1[ki] / D[jc1[ki]] * v1[kj]; ki++; kj++; }
          else if (jc1[ki] < jc1[kj]) ki++; else kj++;
        }
      }
      W[(size_t)(i + row_dest_start) * m_W + j + col_dest_start] += alpha * acc;
    }
  }
  #pragma omp target update from(W[0:w_elems])

  std::fill(h_ref.begin(), h_ref.end(), 0.0);
  reference_mdinvmtrans_diag(m, m, h_rs1.data(), h_jc1.data(), h_v1.data(), h_D.data(),
                             0, 0, alpha, h_ref.data());
  for (int i = 0; i < m; i++)
    for (int j = i; j < m; j++)
      if (!close_enough(h_W[(size_t)i * m + j], h_ref[(size_t)i * m + j], 1e-10)) {
        errors++; i = m; break;
      }

  // benchmark
  #pragma omp target teams distribute parallel for thread_limit(128)
  for (size_t k = 0; k < w_elems; k++) W[k] = 0.0;

  auto start = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
    // one iteration per (i,j) row pair (collapsed) instead of per row i
    #pragma omp target teams distribute parallel for collapse(2) thread_limit(128)
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        if (j < i) continue;
        double acc = 0.0;
        if (i == j) {
          for (int k = rs1[i]; k < rs1[i + 1]; k++)
            acc += v1[k] / D[jc1[k]] * v1[k];
        } else {
          int ki = rs1[i], kj = rs1[j];
          while (ki < rs1[i + 1] && kj < rs1[j + 1]) {
            if (jc1[ki] == jc1[kj]) { acc += v1[ki] / D[jc1[ki]] * v1[kj]; ki++; kj++; }
            else if (jc1[ki] < jc1[kj]) ki++; else kj++;
          }
        }
        W[(size_t)(i + row_dest_start) * m_W + j + col_dest_start] += alpha * acc;
      }
    }
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of addMDinvMtrans (diag block): %f (us)\n",
         time * 1e-3 / repeat);

  // --- off-diagonal block: W += alpha * J1 D^{-1} J2^T ---------------------

  // host/device correctness check before timing
  #pragma omp target teams distribute parallel for thread_limit(128)
  for (size_t k = 0; k < w_elems; k++) W[k] = 0.0;

  #pragma omp target teams distribute parallel for collapse(2) thread_limit(128)
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < m; j++) {
      double acc = 0.0;
      int ki = rs1[i], kj = rs2[j];
      while (ki < rs1[i + 1] && kj < rs2[j + 1]) {
        if (jc1[ki] == jc2[kj]) { acc += v1[ki] / D[jc1[ki]] * v2[kj]; ki++; kj++; }
        else if (jc1[ki] < jc2[kj]) ki++; else kj++;
      }
      W[(size_t)(i + row_dest_start) * m_W + j + col_dest_start] += alpha * acc;
    }
  }
  #pragma omp target update from(W[0:w_elems])

  std::fill(h_ref.begin(), h_ref.end(), 0.0);
  reference_mdinvntrans(m, m, m, h_rs1.data(), h_jc1.data(), h_v1.data(),
                        h_rs2.data(), h_jc2.data(), h_v2.data(), h_D.data(),
                        0, 0, alpha, h_ref.data());
  for (size_t k = 0; k < w_elems; k++)
    if (!close_enough(h_W[k], h_ref[k], 1e-10)) { errors++; break; }

  // benchmark
  #pragma omp target teams distribute parallel for thread_limit(128)
  for (size_t k = 0; k < w_elems; k++) W[k] = 0.0;

  start = std::chrono::steady_clock::now();

  for (int r = 0; r < repeat; r++) {
    #pragma omp target teams distribute parallel for collapse(2) thread_limit(128)
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < m; j++) {
        double acc = 0.0;
        int ki = rs1[i], kj = rs2[j];
        while (ki < rs1[i + 1] && kj < rs2[j + 1]) {
          if (jc1[ki] == jc2[kj]) {
            acc += v1[ki] / D[jc1[ki]] * v2[kj];
            ki++; kj++;
          } else if (jc1[ki] < jc2[kj]) {
            ki++;
          } else {
            kj++;
          }
        }
        W[(size_t)(i + row_dest_start) * m_W + j + col_dest_start] += alpha * acc;
      }
    }
  }

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of addMDinvNtrans (off-diag):   %f (us)\n",
         time * 1e-3 / repeat);

  #pragma omp target exit data map(delete: rs1[0:m+1], jc1[0:nnz], v1[0:nnz], \
                                           rs2[0:m+1], jc2[0:nnz], v2[0:nnz], \
                                           D[0:nx], W[0:w_elems])

  printf("%s\n", errors == 0 ? "PASS" : "FAIL");

  return 0;
}
