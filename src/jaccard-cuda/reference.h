#ifndef JACCARD_REFERENCE_H
#define JACCARD_REFERENCE_H

#include <cmath>
#include <cstdio>
#include <vector>

// Host reference of the Jaccard weights computed by the kernels.
// Vertex weights are those assigned by fill<weighted>: (c+1)/e, or 1.
template <bool weighted, typename T>
bool verify_weights (const int n, const int e,
    const int* csr_ptr, const int* csr_ind, const T* csr_val,
    const T gamma, const T* weight_j)
{
  auto vertex_weight = [e](int c) -> double {
    return weighted ? (double)((T)(c+1)/e) : 1.0;
  };

  std::vector<double> work(n);
  for (int row = 0; row < n; row++) {
    double sum = 0.0;
    for (int k = csr_ptr[row]; k < csr_ptr[row+1]; k++)
      sum += vertex_weight(csr_ind[k]);
    work[row] = sum;
  }

  int errors = 0;
  for (int row = 0; row < n; row++) {
    for (int j = csr_ptr[row]; j < csr_ptr[row+1]; j++) {
      int col = csr_ind[j];
      double wi = 0.0;
      int a = csr_ptr[row], a_end = csr_ptr[row+1];
      int b = csr_ptr[col], b_end = csr_ptr[col+1];
      while (a < a_end && b < b_end) {
        if (csr_ind[a] == csr_ind[b]) {
          wi += vertex_weight(csr_ind[a]);
          a++; b++;
        } else if (csr_ind[a] < csr_ind[b]) {
          a++;
        } else {
          b++;
        }
      }
      double ws = work[row] + work[col];
      double ref = ((double)gamma * csr_val[j]) * (wi / (ws - wi));
      double err = std::fabs(weight_j[j] - ref);
      // the device sums in a different order and precision
      if (!(err <= 1e-6 || err <= 1e-4 * std::fabs(ref))) {
        if (errors++ < 10)
          printf("Mismatch at %d: %f (device) != %f (host)\n", j, weight_j[j], ref);
      }
    }
  }
  return errors == 0;
}

#endif
