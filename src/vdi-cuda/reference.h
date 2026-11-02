// CPU reference for the Video Delta inverse. Storage is row-major.
//
// today_reference is the four torch calls, in the caller's type (fp32 when
// the benchmark compares the "today" line):
//   L    = cholesky(A + I)          the +I is inside cholesky_lower
//   Linv = solve_triangular(L, I, upper=false, left=true)
//   inv  = Linv^T @ Linv
//   T    = diag(alpha) * inv
//   J    = B @ inv
//
// fp64_inv_reference is the acceptance value: a general inverse of I+A
// (partial-pivoting Gauss-Jordan, standing in for torch.linalg.inv), then
// the same epilogue. It is not the Cholesky path. Pass/fail against this
// result is a Frobenius relative error of 1e-6 on both T and J.

#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <thread>
#include <vector>

template <typename Fn>
void parallel_for(int n, Fn fn) {
  int nt = static_cast<int>(std::thread::hardware_concurrency());
  if (nt < 1) nt = 1;
  if (nt > n) nt = n;
  std::atomic<int> next{0};
  std::vector<std::thread> pool;
  pool.reserve(nt);
  for (int t = 0; t < nt; ++t) {
    pool.emplace_back([&] {
      for (;;) {
        const int i = next.fetch_add(1, std::memory_order_relaxed);
        if (i >= n) break;
        fn(i);
      }
    });
  }
  for (auto& th : pool) th.join();
}

// Lower Cholesky of A+I. The caller passes A; the identity is added on the
// pivot before the dot of the previous column. L's upper triangle is left 0.
template <typename T>
void cholesky_lower(const T* A, T* L, int n) {
  std::fill(L, L + static_cast<size_t>(n) * n, T(0));
  for (int k = 0; k < n; ++k) {
    T sum = A[k * n + k] + T(1);
    for (int j = 0; j < k; ++j) sum -= L[k * n + j] * L[k * n + j];
    L[k * n + k] = std::sqrt(sum);
    for (int i = k + 1; i < n; ++i) {
      T s = A[i * n + k];
      for (int j = 0; j < k; ++j) s -= L[i * n + j] * L[k * n + j];
      L[i * n + k] = s / L[k * n + k];
    }
  }
}

// L X = B with L lower triangular. This is solve_triangular(..., upper=false, left=true).
template <typename T>
void solve_triangular_lower(const T* L, const T* B, T* X, int n) {
  for (int c = 0; c < n; ++c) {
    for (int i = 0; i < n; ++i) {
      T s = B[i * n + c];
      for (int j = 0; j < i; ++j) s -= L[i * n + j] * X[j * n + c];
      X[i * n + c] = s / L[i * n + i];
    }
  }
}

// Dot in independent panels of 8, then a short sum of those panels. One
// dependent chain of 128 fp32 products misses 1e-6 on J at beta-scale 50
// and 100. In fp64 the same split is just the epilogue of the acceptance
// inverse and is not what limits that comparison.
template <typename T>
T dot_panels(const T* x, int xstride, const T* y, int ystride, int begin, int end) {
  T acc = T(0);
  int k = begin;
  for (; k + 7 < end; k += 8) {
    T s0 = x[k * xstride] * y[k * ystride];
    T s1 = x[(k + 1) * xstride] * y[(k + 1) * ystride];
    T s2 = x[(k + 2) * xstride] * y[(k + 2) * ystride];
    T s3 = x[(k + 3) * xstride] * y[(k + 3) * ystride];
    T s4 = x[(k + 4) * xstride] * y[(k + 4) * ystride];
    T s5 = x[(k + 5) * xstride] * y[(k + 5) * ystride];
    T s6 = x[(k + 6) * xstride] * y[(k + 6) * ystride];
    T s7 = x[(k + 7) * xstride] * y[(k + 7) * ystride];
    acc += (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7);
  }
  for (; k < end; ++k) acc += x[k * xstride] * y[k * ystride];
  return acc;
}

// Inv = Linv^T @ Linv. Linv is lower triangular, so the dot for element
// (r, c) starts at k = max(r, c): Linv[k, r] is zero for k < r and
// Linv[k, c] is zero for k < c. Each element is computed on its own, including
// both triangles, the way a GEMM would, rather than mirroring one of them.
template <typename T>
void gemm_lt_l(const T* Linv, T* Inv, int n) {
  for (int r = 0; r < n; ++r) {
    for (int c = 0; c < n; ++c) {
      const int k0 = r > c ? r : c;
      Inv[r * n + c] = dot_panels(Linv + r, n, Linv + c, n, k0, n);
    }
  }
}

// T = diag(alpha) * Inv, J = B * Inv. J uses dot_panels so the fp32 four-call
// path meets the same 1e-6 bar as the device kernels.
template <typename T>
void apply_epilogue(const T* B, const T* alpha, const T* Inv, T* Tout, T* Jout,
                    int n) {
  for (int r = 0; r < n; ++r) {
    const T ar = alpha[r];
    const T* brow = B + static_cast<size_t>(r) * n;
    for (int c = 0; c < n; ++c) {
      Tout[r * n + c] = ar * Inv[r * n + c];
      Jout[r * n + c] = dot_panels(brow, 1, Inv + c, n, 0, n);
    }
  }
}

template <typename T>
void today_reference(const T* A, const T* B, const T* alpha, T* Tout, T* Jout,
                     int n) {
  const size_t nn = static_cast<size_t>(n) * n;
  std::vector<T> L(nn), Eye(nn, T(0)), Linv(nn), Inv(nn);
  for (int i = 0; i < n; ++i) Eye[static_cast<size_t>(i) * n + i] = T(1);
  cholesky_lower(A, L.data(), n);
  solve_triangular_lower(L.data(), Eye.data(), Linv.data(), n);
  gemm_lt_l(Linv.data(), Inv.data(), n);
  apply_epilogue(B, alpha, Inv.data(), Tout, Jout, n);
}

template <typename T>
void today_batch(const T* A, const T* B, const T* alpha, T* Tout, T* Jout,
                 int nmat, int n) {
  parallel_for(nmat, [&](int m) {
    const size_t nn = static_cast<size_t>(n) * n;
    today_reference(A + m * nn, B + m * nn, alpha + static_cast<size_t>(m) * n,
                    Tout + m * nn, Jout + m * nn, n);
  });
}

// fp64 partial-pivoting Gauss-Jordan. This is the acceptance inverse, not
// the Cholesky path above. Returns false when a pivot is below 1e-18.
inline bool invert_gauss(const double* M, std::vector<double>& inv, int n) {
  std::vector<double> a(static_cast<size_t>(n) * 2 * n, 0.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) a[i * 2 * n + j] = M[i * n + j];
    a[i * 2 * n + n + i] = 1.0;
  }
  for (int k = 0; k < n; ++k) {
    int piv = k;
    for (int i = k + 1; i < n; ++i)
      if (std::fabs(a[i * 2 * n + k]) > std::fabs(a[piv * 2 * n + k])) piv = i;
    if (std::fabs(a[piv * 2 * n + k]) < 1e-18) return false;
    if (piv != k)
      for (int j = k; j < 2 * n; ++j)
        std::swap(a[k * 2 * n + j], a[piv * 2 * n + j]);
    const double diag = a[k * 2 * n + k];
    for (int j = k; j < 2 * n; ++j) a[k * 2 * n + j] /= diag;
    for (int i = 0; i < n; ++i) {
      if (i == k) continue;
      const double f = a[i * 2 * n + k];
      for (int j = k; j < 2 * n; ++j) a[i * 2 * n + j] -= f * a[k * 2 * n + j];
    }
  }
  inv.resize(static_cast<size_t>(n) * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) inv[i * n + j] = a[i * 2 * n + n + j];
  return true;
}

inline bool fp64_inv_reference(const double* A, const double* B, const double* alpha,
                               double* Tout, double* Jout, int n) {
  std::vector<double> M(static_cast<size_t>(n) * n), Inv;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) M[i * n + j] = A[i * n + j];
    M[i * n + i] += 1.0;
  }
  if (!invert_gauss(M.data(), Inv, n)) return false;
  apply_epilogue(B, alpha, Inv.data(), Tout, Jout, n);
  return true;
}

inline bool fp64_batch(const double* A, const double* B, const double* alpha,
                       double* Tout, double* Jout, int nmat, int n) {
  std::atomic<int> failed{0};
  parallel_for(nmat, [&](int m) {
    const size_t nn = static_cast<size_t>(n) * n;
    if (!fp64_inv_reference(A + m * nn, B + m * nn,
                            alpha + static_cast<size_t>(m) * n,
                            Tout + m * nn, Jout + m * nn, n)) {
      failed.store(1, std::memory_order_relaxed);
    }
  });
  return failed.load(std::memory_order_relaxed) == 0;
}
