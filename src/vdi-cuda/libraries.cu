// cuSOLVER and cuBLAS are column-major. A is symmetric and stored in full,
// so the same bytes are a valid column-major copy: element (i, j) in
// row-major memory is element (j, i) in column-major, and those are equal.
// The diagonal address is the same in both layouts, so adding 1 forms I+A
// either way. potrf(LOWER) then writes the column-major lower triangle,
// which occupies the row-major upper triangle including the diagonal. The
// leftover row-major lower triangle is original A and is never read again;
// trsm uses only the column-major factor.
//
// B is not symmetric. Reading its row-major buffer as column-major yields B^T.
// The J GEMM computes column-major C = inv * B^T. Column-major C[i, j] lives
// at the address of row-major element (j, i), and
// C[i, j] = sum_k inv[i, k] * B[j, k] = (B * inv)[j, i], so the bytes are
// row-major J. inv is symmetric, so column-major Linv^T * Linv is already
// the row-major inverse.
//
// There is no strided trsm in this cuBLAS. potrf and trsm take device arrays
// of pointers; the two products use strided GEMM. potrfBatched has no
// workspace argument.

#include "libraries.h"
#include "library_batch.cuh"

#include <cublas_v2.h>
#include <cusolverDn.h>

#include <cstdio>
#include <vector>

namespace {

constexpr int kDim = 128;

// Workspace for one batch size. workA is the potrf input (A+I, then L).
// eye is a permanent identity; workI is recopied from it every launch because
// trsm overwrites the right-hand side with Linv. workInv holds Linv^T Linv.
// Aptr and Iptr are device pointer arrays, one pointer per matrix.
struct State {
  int nmat = 0;
  float* workA = nullptr;
  float* workI = nullptr;
  float* eye = nullptr;
  float* workInv = nullptr;
  float** Aptr = nullptr;
  float** Iptr = nullptr;
  int* info = nullptr;
  cusolverDnHandle_t sol = nullptr;
  cublasHandle_t blas = nullptr;
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
  auto alloc = [&](float** p) {
    return cudaMalloc(p, bytes) == cudaSuccess;
  };
  if (!alloc(&g.workA) || !alloc(&g.workI) || !alloc(&g.eye) || !alloc(&g.workInv) ||
      cudaMalloc(&g.Aptr, static_cast<size_t>(nmat) * sizeof(float*)) != cudaSuccess ||
      cudaMalloc(&g.Iptr, static_cast<size_t>(nmat) * sizeof(float*)) != cudaSuccess ||
      cudaMalloc(&g.info, static_cast<size_t>(nmat) * sizeof(int)) != cudaSuccess) {
    fail("cudaMalloc", 0);
    libraries_shutdown();
    return 1;
  }

  const int threads = 256;
  const int blocks = static_cast<int>((nn * static_cast<size_t>(nmat) + threads - 1) / threads);
  vdn_fill_identity<<<blocks, threads>>>(g.eye, kDim, nmat);
  if (cudaDeviceSynchronize() != cudaSuccess) {
    fail("fill_identity", 0);
    libraries_shutdown();
    return 1;
  }

  std::vector<float*> ap(nmat), ip(nmat);
  for (int b = 0; b < nmat; ++b) {
    ap[b] = g.workA + static_cast<size_t>(b) * nn;
    ip[b] = g.workI + static_cast<size_t>(b) * nn;
  }
  if (cudaMemcpy(g.Aptr, ap.data(), static_cast<size_t>(nmat) * sizeof(float*),
                 cudaMemcpyHostToDevice) != cudaSuccess ||
      cudaMemcpy(g.Iptr, ip.data(), static_cast<size_t>(nmat) * sizeof(float*),
                 cudaMemcpyHostToDevice) != cudaSuccess) {
    fail("pointer upload", 0);
    libraries_shutdown();
    return 1;
  }

  if (cusolverDnCreate(&g.sol) != CUSOLVER_STATUS_SUCCESS) {
    fail("cusolverDnCreate", 0);
    libraries_shutdown();
    return 1;
  }
  if (cublasCreate(&g.blas) != CUBLAS_STATUS_SUCCESS) {
    fail("cublasCreate", 0);
    libraries_shutdown();
    return 1;
  }
  // Pedantic mode keeps the fp32 GEMMs in fp32. The default Hopper mode
  // contracts them in TF32, which the model authors found destroys the
  // conditioning of I+A.
  if (cublasSetMathMode(g.blas, CUBLAS_PEDANTIC_MATH) != CUBLAS_STATUS_SUCCESS) {
    fail("cublasSetMathMode", 0);
    libraries_shutdown();
    return 1;
  }
  g.nmat = nmat;
  return 0;
}

void libraries_shutdown() {
  if (g.blas) cublasDestroy(g.blas);
  if (g.sol) cusolverDnDestroy(g.sol);
  cudaFree(g.workA);
  cudaFree(g.workI);
  cudaFree(g.eye);
  cudaFree(g.workInv);
  cudaFree(g.Aptr);
  cudaFree(g.Iptr);
  cudaFree(g.info);
  g = State{};
}

int libraries_launch(const float* A, const float* B, const float* alpha,
                     float* T, float* J, int nmat, int check_info) {
  if (nmat != g.nmat || !g.sol || !g.blas) {
    fail("launch before init", nmat);
    return 1;
  }
  // The four calls live in library_batch.cuh. nullptr is the default stream.
  // check_info synchronizes that stream; timed calls pass 0.
  return vdn_library_epilogue(g.sol, g.blas, static_cast<cudaStream_t>(nullptr), g.workA, g.workI,
                              g.eye, g.workInv, g.Aptr, g.Iptr, g.info, A, B, alpha, T, J, nmat,
                              check_info, "libraries");
}
