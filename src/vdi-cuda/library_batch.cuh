#pragma once

// One copy of the four library calls, shared by the CUDA driver, the HIP
// driver, and the SYCL vendor wrappers:
//   L    = potrf(A + I)
//   Linv = trsm(L, I)
//   inv  = Linv^T * Linv
//   J    = B * inv, in panels of 32
//   T    = diag(alpha) * inv
// A single k=128 GEMM for J misses 1e-6 at beta-scale 50. Panels of 32 pass.
// The handles must already be in fp32 (pedantic on CUDA, default math on HIP).

#include <cstdio>
#include <vector>

constexpr int kVdnDim = 128;
constexpr int kVdnPanel = 32;

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include <hipsolver/hipsolver.h>
using vdn_stream_t = hipStream_t;
using vdn_blas_t = hipblasHandle_t;
using vdn_sol_t = hipsolverHandle_t;
#define VDN_SOL_OK HIPSOLVER_STATUS_SUCCESS
#define VDN_BLAS_OK HIPBLAS_STATUS_SUCCESS
#define VDN_POTRF hipsolverDnSpotrfBatched
#define VDN_TRSM hipblasStrsmBatched
#define VDN_GEMM hipblasSgemmStridedBatched
#define VDN_SET_SOL_STREAM hipsolverSetStream
#define VDN_SET_BLAS_STREAM hipblasSetStream
#define VDN_FILL HIPBLAS_FILL_MODE_LOWER
#define VDN_SIDE HIPBLAS_SIDE_LEFT
#define VDN_OP_N HIPBLAS_OP_N
#define VDN_OP_T HIPBLAS_OP_T
#define VDN_DIAG HIPBLAS_DIAG_NON_UNIT
#define VDN_MEMCPY_ASYNC hipMemcpyAsync
#define VDN_MEMCPY hipMemcpy
#define VDN_STREAM_SYNC hipStreamSynchronize
#define VDN_D2D hipMemcpyDeviceToDevice
#define VDN_D2H hipMemcpyDeviceToHost
#else
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
using vdn_stream_t = cudaStream_t;
using vdn_blas_t = cublasHandle_t;
using vdn_sol_t = cusolverDnHandle_t;
#define VDN_SOL_OK CUSOLVER_STATUS_SUCCESS
#define VDN_BLAS_OK CUBLAS_STATUS_SUCCESS
#define VDN_POTRF cusolverDnSpotrfBatched
#define VDN_TRSM cublasStrsmBatched
#define VDN_GEMM cublasSgemmStridedBatched
#define VDN_SET_SOL_STREAM cusolverDnSetStream
#define VDN_SET_BLAS_STREAM cublasSetStream
#define VDN_FILL CUBLAS_FILL_MODE_LOWER
#define VDN_SIDE CUBLAS_SIDE_LEFT
#define VDN_OP_N CUBLAS_OP_N
#define VDN_OP_T CUBLAS_OP_T
#define VDN_DIAG CUBLAS_DIAG_NON_UNIT
#define VDN_MEMCPY_ASYNC cudaMemcpyAsync
#define VDN_MEMCPY cudaMemcpy
#define VDN_STREAM_SYNC cudaStreamSynchronize
#define VDN_D2D cudaMemcpyDeviceToDevice
#define VDN_D2H cudaMemcpyDeviceToHost
#endif

__global__ void vdn_add_identity(float* A, int n, int nmat) {
  const int m = blockIdx.x;
  const int i = threadIdx.x;
  if (m < nmat && i < n)
    A[(static_cast<size_t>(m) * n + i) * n + i] += 1.0f;
}

__global__ void vdn_fill_identity(float* Eye, int n, int nmat) {
  const int nn = n * n;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nmat * nn) return;
  const int e = idx - (idx / nn) * nn;
  const int r = e / n;
  const int c = e - r * n;
  Eye[idx] = (r == c) ? 1.0f : 0.0f;
}

__global__ void vdn_scale_rows(float* T, const float* Inv, const float* alpha, int n,
                              int nmat) {
  const int nn = n * n;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= nmat * nn) return;
  const int m = idx / nn;
  const int r = (idx - m * nn) / n;
  T[idx] = Inv[idx] * alpha[static_cast<size_t>(m) * n + r];
}

// stream nullptr is the default stream. check_info synchronizes that stream
// and rejects a potrf info value other than 0. Returns 0 on success.
inline int vdn_library_epilogue(vdn_sol_t sol, vdn_blas_t blas, vdn_stream_t stream,
                                float* workA, float* workI, float* eye, float* workInv,
                                float** Aptr, float** Iptr, int* info, const float* A,
                                const float* B, const float* alpha, float* T, float* J,
                                int nmat, int check_info, const char* who) {
  auto fail = [&](const char* what, int code) {
    fprintf(stderr, "%s: %s failed (%d)\n", who, what, code);
    return 1;
  };
  if (VDN_SET_SOL_STREAM(sol, stream) != VDN_SOL_OK ||
      VDN_SET_BLAS_STREAM(blas, stream) != VDN_BLAS_OK)
    return fail("set stream", 0);
  const size_t nn = static_cast<size_t>(kVdnDim) * kVdnDim;
  const size_t bytes = nn * static_cast<size_t>(nmat) * sizeof(float);
  if (VDN_MEMCPY_ASYNC(workA, A, bytes, VDN_D2D, stream) != 0) return fail("copy A", 0);
  vdn_add_identity<<<nmat, kVdnDim, 0, stream>>>(workA, kVdnDim, nmat);
  if (VDN_POTRF(sol, VDN_FILL, kVdnDim, Aptr, kVdnDim, info, nmat) != VDN_SOL_OK)
    return fail("potrf", 0);
  if (VDN_MEMCPY_ASYNC(workI, eye, bytes, VDN_D2D, stream) != 0) return fail("restore I", 0);
  const float one = 1.0f;
  const float zero = 0.0f;
  if (VDN_TRSM(blas, VDN_SIDE, VDN_FILL, VDN_OP_N, VDN_DIAG, kVdnDim, kVdnDim, &one,
               reinterpret_cast<const float* const*>(Aptr), kVdnDim, Iptr, kVdnDim,
               nmat) != VDN_BLAS_OK)
    return fail("trsm", 0);
  const long long stride = static_cast<long long>(nn);
  if (VDN_GEMM(blas, VDN_OP_T, VDN_OP_N, kVdnDim, kVdnDim, kVdnDim, &one, workI, kVdnDim, stride,
               workI, kVdnDim, stride, &zero, workInv, kVdnDim, stride, nmat) != VDN_BLAS_OK)
    return fail("gemm inv", 0);
  for (int k0 = 0; k0 < kVdnDim; k0 += kVdnPanel) {
    const float* beta = (k0 == 0) ? &zero : &one;
    if (VDN_GEMM(blas, VDN_OP_N, VDN_OP_N, kVdnDim, kVdnDim, kVdnPanel, &one,
                 workInv + static_cast<size_t>(k0) * kVdnDim, kVdnDim, stride, B + k0, kVdnDim,
                 stride, beta, J, kVdnDim, stride, nmat) != VDN_BLAS_OK)
      return fail("gemm J", 0);
  }
  const int threads = 256;
  const int blocks =
      static_cast<int>((nn * static_cast<size_t>(nmat) + threads - 1) / threads);
  vdn_scale_rows<<<blocks, threads, 0, stream>>>(T, workInv, alpha, kVdnDim, nmat);
  if (check_info) {
    if (VDN_STREAM_SYNC(stream) != 0) return fail("sync", 0);
    std::vector<int> hinfo(nmat);
    if (VDN_MEMCPY(hinfo.data(), info, static_cast<size_t>(nmat) * sizeof(int), VDN_D2H) != 0)
      return fail("copy info", 0);
    int nbad = 0;
    for (int v : hinfo)
      if (v != 0) ++nbad;
    if (nbad) return fail("potrf info", nbad);
  }
  return 0;
}
