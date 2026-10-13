// oneMKL on the caller's in-order SYCL queue. Same four calls as the CUDA
// library path: potrf, trsm, inv = Linv^T * Linv, then J = B * inv
// in panels of 32. All fp32. trsm and both gemms pass compute_mode::standard.
//
// A is symmetric, so the row-major buffer is also column-major A. potrf
// LOWER overwrites that lower triangle. trsm solves L X = I. The inverse
// is symmetric, so the column-major product is the row-major inverse.
// J is column-major inv[:, k0:k0+32] * B_col[k0:k0+32, :], which stores
// row-major B * inv. A single k=128 GEMM for J misses 1e-6 at beta 50.
//
// The Makefile always compiles this path in (-DWDN_ONEMKL): -qmkl=parallel
// on the Intel compiler, and the oneMKL interface for the CUDA and HIP targets.

#include "libraries.h"

#include <cstdio>

#ifndef WDN_ONEMKL

int libraries_init(sycl::queue&, int) {
  fprintf(stderr, "libraries: oneMKL is not in this binary\n");
  return 2;
}

void libraries_shutdown() {}

int libraries_launch(sycl::queue&, const float*, const float*, const float*, float*, float*,
                     int, int) {
  return 2;
}

#else

#include <oneapi/mkl/blas.hpp>
#include <oneapi/mkl/lapack.hpp>

#include <memory>
#include <vector>

namespace {

constexpr int kDim = 128;
constexpr int kPanel = 32;

struct State {
  sycl::queue q;
  int nmat = 0;
  float* workA = nullptr;
  float* workI = nullptr;
  float* eye = nullptr;
  float* workInv = nullptr;
  float* scratch = nullptr;
  int* info = nullptr;
  std::int64_t scratch_n = 0;
};

State* g = nullptr;
int g_ready = 0;  // 1 initialized, 2 no device image

void fail(const char* what) { fprintf(stderr, "oneMKL: %s failed\n", what); }

void free_usm(State& st) {
  sycl::queue& q = st.q;
  if (st.workA) sycl::free(st.workA, q);
  if (st.workI) sycl::free(st.workI, q);
  if (st.eye) sycl::free(st.eye, q);
  if (st.workInv) sycl::free(st.workInv, q);
  if (st.scratch) sycl::free(st.scratch, q);
  if (st.info) sycl::free(st.info, q);
  st.workA = st.workI = st.eye = st.workInv = st.scratch = nullptr;
  st.info = nullptr;
}

void add_identity(sycl::queue& q, float* A, int nmat) {
  const int n = kDim;
  q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::range<1>(static_cast<size_t>(nmat) * n), [=](sycl::id<1> id) {
      const size_t i = id % n;
      const size_t m = id / n;
      A[(m * n + i) * n + i] += 1.0f;
    });
  });
}

void scale_rows(sycl::queue& q, float* T, const float* Inv, const float* alpha,
                int nmat) {
  const int n = kDim;
  const size_t nn = static_cast<size_t>(n) * n;
  q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::range<1>(nn * nmat), [=](sycl::id<1> idx) {
      const size_t m = idx / nn;
      const size_t r = (idx % nn) / n;
      T[idx] = Inv[idx] * alpha[m * n + r];
    });
  });
}

// potrf has no info array in the strided oneMKL entry. A non-positive
// diagonal is the failure signal the cuSOLVER info check is standing in for.
void check_diag(sycl::queue& q, const float* A, int* info, int nmat) {
  const int n = kDim;
  q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::range<1>(nmat), [=](sycl::id<1> m) {
      int bad = 0;
      for (int i = 0; i < n; ++i) {
        const float d = A[(m * n + i) * n + i];
        if (!(d > 0.0f) || d != d) bad = 1;
      }
      info[m] = bad;
    });
  });
}

int probe_device(sycl::queue& q) {
  float* a = nullptr;
  float* scratch = nullptr;
  try {
    a = sycl::malloc_device<float>(1, q);
    if (!a) return 1;
    q.single_task([=]() { a[0] = 1.0f; });
    const std::int64_t nscratch = oneapi::mkl::lapack::potrf_batch_scratchpad_size<float>(
        q, oneapi::mkl::uplo::lower, 1, 1, 1, 1);
    if (nscratch > 0) scratch = sycl::malloc_device<float>(static_cast<size_t>(nscratch), q);
    oneapi::mkl::lapack::potrf_batch(q, oneapi::mkl::uplo::lower, 1, a, 1, 1, 1, scratch,
                                    nscratch);
    q.wait_and_throw();
    sycl::free(a, q);
    if (scratch) sycl::free(scratch, q);
    return 0;
  } catch (const std::exception&) {
    if (a) sycl::free(a, q);
    if (scratch) sycl::free(scratch, q);
    fprintf(stderr, "oneMKL: no kernel image for this GPU\n");
    return 2;
  }
}

}  // namespace

int libraries_init(sycl::queue& q, int nmat) {
  libraries_shutdown();
  if (nmat < 1) return 1;
  const int probed = probe_device(q);
  if (probed != 0) {
    g_ready = probed;
    return probed;
  }
  std::unique_ptr<State> st;
  try {
    st = std::make_unique<State>();
    st->q = q;
    st->nmat = nmat;
    const size_t nn = static_cast<size_t>(kDim) * kDim;
    const size_t ntot = nn * static_cast<size_t>(nmat);
    st->workA = sycl::malloc_device<float>(ntot, q);
    st->workI = sycl::malloc_device<float>(ntot, q);
    st->eye = sycl::malloc_device<float>(ntot, q);
    st->workInv = sycl::malloc_device<float>(ntot, q);
    st->info = sycl::malloc_device<int>(nmat, q);
    if (!st->workA || !st->workI || !st->eye || !st->workInv || !st->info) {
      fail("malloc");
      free_usm(*st);
      return 1;
    }
    st->scratch_n = oneapi::mkl::lapack::potrf_batch_scratchpad_size<float>(
        q, oneapi::mkl::uplo::lower, kDim, kDim, static_cast<std::int64_t>(nn), nmat);
    if (st->scratch_n < 0) {
      fail("scratchpad size");
      free_usm(*st);
      return 1;
    }
    if (st->scratch_n > 0) {
      st->scratch = sycl::malloc_device<float>(static_cast<size_t>(st->scratch_n), q);
      if (!st->scratch) {
        fail("scratchpad malloc");
        free_usm(*st);
        return 1;
      }
    }
    q.submit([&](sycl::handler& h) {
       float* eye = st->eye;
       h.parallel_for(sycl::range<1>(ntot), [=](sycl::id<1> idx) {
         const size_t e = idx % nn;
         eye[idx] = (e / kDim == e % kDim) ? 1.0f : 0.0f;
       });
     });
    q.wait_and_throw();
    g = st.release();
    g_ready = 1;
    return 0;
  } catch (const std::exception& ex) {
    fprintf(stderr, "oneMKL init: %s\n", ex.what());
    if (st) free_usm(*st);
    libraries_shutdown();
    return 1;
  }
}

void libraries_shutdown() {
  g_ready = 0;
  if (!g) return;
  free_usm(*g);
  delete g;
  g = nullptr;
}

int libraries_launch(sycl::queue& q, const float* A, const float* B, const float* alpha,
                     float* T, float* J, int nmat, int check_info) {
  if (g_ready == 2) return 2;
  if (!g || nmat != g->nmat) {
    fail("launch before init");
    return 1;
  }
  try {
    const std::int64_t n = kDim;
    const std::int64_t stride = n * n;
    const size_t bytes = static_cast<size_t>(stride) * static_cast<size_t>(nmat) * sizeof(float);
    const float one = 1.0f;
    const float zero = 0.0f;
    namespace mkl = oneapi::mkl;
    q.memcpy(g->workA, A, bytes);
    add_identity(q, g->workA, nmat);
    mkl::lapack::potrf_batch(q, mkl::uplo::lower, n, g->workA, n, stride, nmat, g->scratch,
                             g->scratch_n);
    q.memcpy(g->workI, g->eye, bytes);
    mkl::blas::trsm_batch(q, mkl::side::left, mkl::uplo::lower, mkl::transpose::nontrans,
                          mkl::diag::nonunit, n, n, one, g->workA, n, stride, g->workI, n,
                          stride, nmat, mkl::blas::compute_mode::standard);
    mkl::blas::gemm_batch(q, mkl::transpose::trans, mkl::transpose::nontrans, n, n, n, one,
                          g->workI, n, stride, g->workI, n, stride, zero, g->workInv, n, stride,
                          nmat, mkl::blas::compute_mode::standard);
    for (int k0 = 0; k0 < kDim; k0 += kPanel) {
      const float beta = (k0 == 0) ? zero : one;
      mkl::blas::gemm_batch(
          q, mkl::transpose::nontrans, mkl::transpose::nontrans, n, n, kPanel, one,
          g->workInv + static_cast<size_t>(k0) * kDim, n, stride, B + k0, n, stride, beta, J, n,
          stride, nmat, mkl::blas::compute_mode::standard);
    }
    scale_rows(q, T, g->workInv, alpha, nmat);
    if (check_info) {
      check_diag(q, g->workA, g->info, nmat);
      std::vector<int> hinfo(nmat);
      q.memcpy(hinfo.data(), g->info, static_cast<size_t>(nmat) * sizeof(int));
      q.wait_and_throw();
      for (int v : hinfo) {
        if (v != 0) {
          fail("potrf diagonal");
          return 1;
        }
      }
    }
    return 0;
  } catch (const std::exception& ex) {
    fprintf(stderr, "oneMKL launch: %s\n", ex.what());
    return 1;
  }
}

#endif  // WDN_ONEMKL
