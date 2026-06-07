///////////////////////////////////////////////////////////////////////////////
// Header for common includes and utility functions
///////////////////////////////////////////////////////////////////////////////

#ifndef COMMON_H
#define COMMON_H

///////////////////////////////////////////////////////////////////////////////
// Common includes
///////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <memory.h>
#include <math.h>
#include <sycl/sycl.hpp>

///////////////////////////////////////////////////////////////////////////////
// Common constants
///////////////////////////////////////////////////////////////////////////////
const int StrideAlignment = 256;

///////////////////////////////////////////////////////////////////////////////
// Common functions
///////////////////////////////////////////////////////////////////////////////

// Align up n to the nearest multiple of m
inline int iAlignUp(int n, int m = StrideAlignment) {
  int mod = n % m;

  if (mod)
    return n + m - mod;
  else
    return n;
}

// round up n/m
inline int iDivUp(int n, int m) { return (n + m - 1) / m; }

// swap two values
template <typename T>
inline void Swap(T &a, T &b) {
  T t = a;
  a = b;
  b = t;
}

#define CHECK_ERROR(expr)                   \
  [&]() {                                   \
    try {                                   \
      expr;                                 \
      return 0;                             \
    } catch (std::exception const &e) {     \
      std::cerr << e.what() << std::endl;   \
      return -1;                            \
    }                                       \
  }()

#endif
