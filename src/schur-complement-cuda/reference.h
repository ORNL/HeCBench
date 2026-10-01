#ifndef REFERENCE_H
#define REFERENCE_H

#include <stdlib.h>
#include <math.h>
#include <vector>
#include <algorithm>

// https://github.com/ORNL/hiop
//
//   addMDinvMtransToDiagBlockOfSymDeMatUTri :  W += alpha * J * D^{-1} * J^T
//   addMDinvNtransToSymDeMatUTri            :  W += alpha * J1 * D^{-1} * J2^T
//
// J is a sparse matrix stored CSR-style with per-row sorted column indices,
// D is a diagonal (stored as a vector), and W is a dense matrix into whose
// upper triangle the (symmetric) result is scattered.

// Build a CSR matrix with `m` rows, `nx` columns and exactly `nnz_row`
// nonzeros per row (column indices sorted ascending within each row).
static void gen_csr(int m, int nx, int nnz_row, unsigned seed,
                    std::vector<int>& row_start,
                    std::vector<int>& jcol,
                    std::vector<double>& values)
{
  if (nnz_row > nx) nnz_row = nx;
  srand(seed);

  row_start.resize(m + 1);
  jcol.resize((size_t)m * nnz_row);
  values.resize((size_t)m * nnz_row);

  std::vector<int> cols(nx);
  for (int c = 0; c < nx; c++) cols[c] = c;

  row_start[0] = 0;
  for (int i = 0; i < m; i++) {
    // partial Fisher-Yates: pick the first nnz_row distinct columns
    for (int k = 0; k < nnz_row; k++) {
      int r = k + rand() % (nx - k);
      std::swap(cols[k], cols[r]);
    }
    std::sort(cols.begin(), cols.begin() + nnz_row);
    const size_t base = (size_t)i * nnz_row;
    for (int k = 0; k < nnz_row; k++) {
      jcol[base + k] = cols[k];
      values[base + k] = 2.0 * (rand() / (double)RAND_MAX) - 1.0;
    }
    row_start[i + 1] = row_start[i] + nnz_row;
  }
}

// D is the diagonal; keep entries well away from zero so 1/D is well behaved.
static void gen_diag(int nx, unsigned seed, std::vector<double>& D)
{
  srand(seed);
  D.resize(nx);
  for (int c = 0; c < nx; c++) D[c] = 0.1 + rand() / (double)RAND_MAX;
}

// W += alpha * J * D^{-1} * J^T  (upper triangle of the m x m diagonal block)
static void reference_mdinvmtrans_diag(int m, int m_W,
                                       const int* row_start, const int* jcol,
                                       const double* values, const double* D,
                                       int row_dest_start, int col_dest_start,
                                       double alpha, double* W)
{
  for (int i = 0; i < m; i++) {
    double acc = 0.0;
    for (int k = row_start[i]; k < row_start[i + 1]; k++)
      acc += values[k] / D[jcol[k]] * values[k];
    W[(size_t)(i + row_dest_start) * m_W + i + col_dest_start] += alpha * acc;

    for (int j = i + 1; j < m; j++) {
      acc = 0.0;
      int ki = row_start[i], kj = row_start[j];
      while (ki < row_start[i + 1] && kj < row_start[j + 1]) {
        if (jcol[ki] == jcol[kj]) {
          acc += values[ki] / D[jcol[ki]] * values[kj];
          ki++; kj++;
        } else if (jcol[ki] < jcol[kj]) {
          ki++;
        } else {
          kj++;
        }
      }
      W[(size_t)(i + row_dest_start) * m_W + j + col_dest_start] += alpha * acc;
    }
  }
}

// W += alpha * J1 * D^{-1} * J2^T  (full m1 x m2 off-diagonal block)
static void reference_mdinvntrans(int m1, int m2, int m_W,
                                  const int* rs1, const int* jc1, const double* v1,
                                  const int* rs2, const int* jc2, const double* v2,
                                  const double* D,
                                  int row_dest_start, int col_dest_start,
                                  double alpha, double* W)
{
  for (int i = 0; i < m1; i++) {
    for (int j = 0; j < m2; j++) {
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

static bool close_enough(double a, double b, double tol)
{
  return fabs(a - b) <= tol * (1.0 + fabs(b));
}

#endif
