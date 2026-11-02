#pragma once

// Library path for the same epilogue as fused.cu, using the four calls from
// the torch reference:
//   L    = potrf(A + I)          cusolverDnSpotrfBatched
//   Linv = trsm(L, I)            cublasStrsmBatched
//   inv  = Linv^T * Linv         cublasSgemmStridedBatched
//   J    = B * inv               same GEMM, k split into panels of 32
//   T    = diag(alpha) * inv
//
// All arithmetic is fp32. The cuBLAS handle is in pedantic mode so Hopper
// does not contract the GEMMs in TF32. A and B are read only; potrf and trsm
// overwrite private copies.
//
// libraries_init(nmat) allocates that workspace. Both calls return 0 on
// success and 1 on failure. check_info copies the potrf status back and fails
// if any matrix did not factor; pass 0 on the timed calls so that sync stays
// outside the measured interval. HIP uses the same four calls. SYCL oneMKL
// does too, and checks a positive diagonal because that potrf has no info array.
int libraries_init(int nmat);
void libraries_shutdown();
int libraries_launch(const float* A, const float* B, const float* alpha,
                     float* T, float* J, int nmat, int check_info);
