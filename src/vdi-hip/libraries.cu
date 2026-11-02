#include "libraries.h"
#include "library_batch.cuh"

#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>

#include <cstdio>
#include <vector>

namespace {

constexpr int kDim = 128;

struct State {
  int nmat = 0;
  float* workA = nullptr;
  float* workI = nullptr;
  float* eye = nullptr;
  float* workInv = nullptr;
  float** Aptr = nullptr;
  float** Iptr = nullptr;
  int* info = nullptr;
  hipsolverHandle_t sol = nullptr;
  hipblasHandle_t blas = nullptr;
};

State g;

void fail(const char* what, int code) {
  fprintf(stderr, "libraries: %s failed (%d)\n", what, code);
}

}  // namespace

int libraries_init(int nmat) {
  if (nmat < 1) {
    fail("nmat", nmat);
    return 1;
  }
  libraries_shutdown();
  const size_t nn = static_cast<size_t>(kDim) * kDim;
  const size_t bytes = nn * static_cast<size_t>(nmat) * sizeof(float);
  auto alloc = [&](float** p) { return hipMalloc(p, bytes) == hipSuccess; };
  if (!alloc(&g.workA) || !alloc(&g.workI) || !alloc(&g.eye) || !alloc(&g.workInv) ||
      hipMalloc(&g.Aptr, static_cast<size_t>(nmat) * sizeof(float*)) != hipSuccess ||
      hipMalloc(&g.Iptr, static_cast<size_t>(nmat) * sizeof(float*)) != hipSuccess ||
      hipMalloc(&g.info, static_cast<size_t>(nmat) * sizeof(int)) != hipSuccess) {
    fail("hipMalloc", 0);
    libraries_shutdown();
    return 1;
  }
  const int threads = 256;
  const int blocks = static_cast<int>((nn * static_cast<size_t>(nmat) + threads - 1) / threads);
  vdn_fill_identity<<<blocks, threads>>>(g.eye, kDim, nmat);
  if (hipDeviceSynchronize() != hipSuccess) {
    fail("fill_identity", 0);
    libraries_shutdown();
    return 1;
  }
  std::vector<float*> ap(nmat), ip(nmat);
  for (int b = 0; b < nmat; ++b) {
    ap[b] = g.workA + static_cast<size_t>(b) * nn;
    ip[b] = g.workI + static_cast<size_t>(b) * nn;
  }
  if (hipMemcpy(g.Aptr, ap.data(), static_cast<size_t>(nmat) * sizeof(float*),
                hipMemcpyHostToDevice) != hipSuccess ||
      hipMemcpy(g.Iptr, ip.data(), static_cast<size_t>(nmat) * sizeof(float*),
                hipMemcpyHostToDevice) != hipSuccess) {
    fail("pointer upload", 0);
    libraries_shutdown();
    return 1;
  }
  if (hipsolverCreate(&g.sol) != HIPSOLVER_STATUS_SUCCESS) {
    fail("hipsolverCreate", 0);
    libraries_shutdown();
    return 1;
  }
  if (hipblasCreate(&g.blas) != HIPBLAS_STATUS_SUCCESS) {
    fail("hipblasCreate", 0);
    libraries_shutdown();
    return 1;
  }
  // DEFAULT_MATH is full fp32. XF32_XDL is the reduced-precision opt-in.
  if (hipblasSetMathMode(g.blas, HIPBLAS_DEFAULT_MATH) != HIPBLAS_STATUS_SUCCESS) {
    fail("hipblasSetMathMode", 0);
    libraries_shutdown();
    return 1;
  }
  g.nmat = nmat;
  return 0;
}

void libraries_shutdown() {
  if (g.blas) hipblasDestroy(g.blas);
  if (g.sol) hipsolverDestroy(g.sol);
  (void)hipFree(g.workA);
  (void)hipFree(g.workI);
  (void)hipFree(g.eye);
  (void)hipFree(g.workInv);
  (void)hipFree(g.Aptr);
  (void)hipFree(g.Iptr);
  (void)hipFree(g.info);
  g = State{};
}

int libraries_launch(const float* A, const float* B, const float* alpha,
                     float* T, float* J, int nmat, int check_info) {
  if (nmat != g.nmat || !g.sol || !g.blas) {
    fail("launch before init", nmat);
    return 1;
  }
  return vdn_library_epilogue(g.sol, g.blas, static_cast<hipStream_t>(nullptr), g.workA, g.workI,
                              g.eye, g.workInv, g.Aptr, g.Iptr, g.info, A, B, alpha, T, J, nmat,
                              check_info, "libraries");
}
