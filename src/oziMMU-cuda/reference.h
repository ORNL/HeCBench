#ifndef REFERENCE_H
#define REFERENCE_H

// Naive host-side double-precision GEMM used as ground truth for accuracy
// checking: C = A * B, with A (m x k), B (k x n), C (m x n), all column-major.
// This is intentionally simple and is only invoked for small problem sizes.
static void reference(const int m, const int n, const int k,
                      const double *A, const double *B, double *C)
{
  for (int j = 0; j < n; j++) {
    for (int i = 0; i < m; i++) {
      double acc = 0.0;
      for (int p = 0; p < k; p++) {
        acc += A[(size_t)p * m + i] * B[(size_t)j * k + p];
      }
      C[(size_t)j * m + i] = acc;
    }
  }
}

#endif
