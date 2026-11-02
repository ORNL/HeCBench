#pragma once

// Library path for the same epilogue as fused.cu, using the four calls from
// the torch reference:
//   L    = potrf(A + I)          hipsolverDnSpotrfBatched
//   Linv = trsm(L, I)            hipblasStrsmBatched
//   inv  = Linv^T * Linv         hipblasSgemmStridedBatched
//   J    = B * inv               same GEMM, k split into panels of 32
//   T    = diag(alpha) * inv
//
// All arithmetic is fp32. rocBLAS has no pedantic mode; HIPBLAS_DEFAULT_MATH
// is fp32 (xf32 is not selected). A and B are read only; potrf and trsm
// overwrite private copies.
//
// libraries_init(nmat) allocates that workspace. Both calls return 0 on
// success and 1 on failure. check_info copies the potrf status back and fails
// if any matrix did not factor; pass 0 on the timed calls so that sync stays
// outside the measured interval. CUDA uses the same four calls. SYCL oneMKL
// does too, and checks a positive diagonal because that potrf has no info array.
int libraries_init(int nmat);
void libraries_shutdown();
int libraries_launch(const float* A, const float* B, const float* alpha,
                     float* T, float* J, int nmat, int check_info);
