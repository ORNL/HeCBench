#pragma once

#include <sycl/sycl.hpp>

// Library path for the same epilogue as fused.cpp, using the four calls from
// the torch reference, in oneMKL on the caller's in-order queue:
//   L    = potrf(A + I)          lapack::potrf_batch
//   Linv = trsm(L, I)            blas::trsm_batch
//   inv  = Linv^T * Linv         blas::gemm_batch
//   J    = B * inv               same GEMM, k split into panels of 32
//   T    = diag(alpha) * inv
//
// All arithmetic is fp32. The BLAS calls pass compute_mode::standard so the
// products are not contracted in TF32. A and B are read only; potrf and trsm
// overwrite private copies.
//
// libraries_init(q, nmat) allocates that workspace. Both calls return 0 on
// success, 1 on failure, and 2 when oneMKL has no kernel image for this
// device or is not in this binary. check_info checks for a positive potrf
// diagonal, because the strided oneMKL potrf has no info array, and
// synchronizes; pass 0 on the timed calls.
int libraries_init(sycl::queue& q, int nmat);
void libraries_shutdown();
int libraries_launch(sycl::queue& q, const float* A, const float* B,
                     const float* alpha, float* T, float* J, int nmat,
                     int check_info);
