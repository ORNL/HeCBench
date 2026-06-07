// Header for common includes and utility functions

#ifndef COMMON_H
#define COMMON_H

#include <math.h>
#include <memory.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

const int StrideAlignment = 256;

inline int iAlignUp(int n, int m = StrideAlignment) {
  int mod = n % m;
  if (mod)
    return n + m - mod;
  return n;
}

inline int iDivUp(int n, int m) { return (n + m - 1) / m; }

template <typename T>
inline void Swap(T &a, T &b) {
  T t = a;
  a = b;
  b = t;
}

#endif
