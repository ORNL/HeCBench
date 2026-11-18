#ifndef JACOBI_REFERENCE_H
#define JACOBI_REFERENCE_H

#include <cmath>
#include <cstdio>

// Check the solution f on the host. The boundary must still hold the initial
// values; only the boundary of f_init is read. The size of a Jacobi update
// does not grow from one step to the next, so one more step on the converged
// interior must not change it by more than the convergence tolerance
// (normalized as in the solver).
static inline bool verify_solution (const int n, const float* f,
                                    const float* f_init, const float tolerance)
{
  int errors = 0;
  double sum = 0.0;
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      const float x = f[i + j * n];
      if (i == 0 || i == n-1 || j == 0 || j == n-1) {
        if (x != f_init[i + j * n]) {
          if (errors++ < 10)
            printf("Boundary (%d, %d): %f != %f\n", i, j, x, f_init[i + j * n]);
        }
      } else {
        float t = 0.25f * (f[(i-1) + j * n] +
                           f[(i+1) + j * n] +
                           f[i + (j-1) * n] +
                           f[i + (j+1) * n]);
        double df = (double)t - x;
        sum += df * df;
      }
    }
  }

  double rms = sqrt(sum / ((double)n * n));
  // allow for rounding in the device's float reduction of the error
  if (!(rms <= 1.01 * tolerance)) {
    printf("Update of one more Jacobi step on the host: %e > %e\n", rms, (double)tolerance);
    errors++;
  }
  return errors == 0;
}

#endif
