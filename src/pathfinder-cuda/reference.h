#ifndef PATHFINDER_REFERENCE_H
#define PATHFINDER_REFERENCE_H

#include <cstring>

// Sequential DP: each cell adds its weight to the cheapest of the
// previous row's left, up, and right neighbors. `out` is the last row.
static inline void PathfinderReference(const int* wall, int rows, int cols, int* out)
{
  int* buf0 = new int[cols];
  int* buf1 = new int[cols];
  memcpy(buf0, wall, sizeof(int) * cols);
  int* src = buf0;
  int* dst = buf1;

  for (int t = 0; t < rows - 1; t++) {
    for (int n = 0; n < cols; n++) {
      int minv = src[n];
      if (n > 0 && src[n - 1] < minv)
        minv = src[n - 1];
      if (n < cols - 1 && src[n + 1] < minv)
        minv = src[n + 1];
      dst[n] = wall[(t + 1) * cols + n] + minv;
    }
    int* tmp = src;
    src = dst;
    dst = tmp;
  }

  memcpy(out, src, sizeof(int) * cols);
  delete[] buf0;
  delete[] buf1;
}

#endif
