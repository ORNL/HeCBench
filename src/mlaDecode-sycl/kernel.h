// oneMKL 3-step MLA decode backend:
//     S = scale * Q . K^T      (GEMM1 via oneMKL gemm_batch, scale = alpha)
//     P = softmax_row(S)       (dedicated SYCL kernel; see softmax.h)
//     O = P . V                (GEMM2 via oneMKL gemm_batch)
//
// The QK scale is folded into GEMM1's alpha.
// oneMKL bf16 GEMM accumulates in fp32; GEMM2 writes its bf16 output directly
// (bf16/bf16/bf16 with fp32 alpha/beta), so no post-GEMM conversion is needed.
//
// column-major encoding of the row-major identities C^T = (A B)^T = B^T A^T
// mirrors the CUDA/cuBLAS argument order exactly.

#include <oneapi/mkl.hpp>
#include "softmax.h"

static void launch(sycl::queue &q, int B, int H, int max_blocks, float scale,
                   const bf16_t *dq, const bf16_t *dkv,
                   bf16_t *dout, float *dlse) {
  const int seqlen = max_blocks * PAGE;
  const size_t rows = (size_t)B * H;

  // Cached scratch: scores S (fp32) and probabilities P (bf16).
  static float  *dS  = nullptr;
  static bf16_t *dP  = nullptr;
  static size_t  capS = 0;
  const size_t needS = rows * seqlen;
  if (needS > capS) {
    if (dS) sycl::free(dS, q);
    if (dP) sycl::free(dP, q);
    dS = sycl::malloc_device<float>(needS, q);
    dP = sycl::malloc_device<bf16_t>(needS, q);
    capS = needS;
  }

  namespace mkl = oneapi::mkl;
  const auto T = mkl::transpose::trans, N = mkl::transpose::nontrans;

  // GEMM1: S[H, s_k] = scale * Q[H, DIM] . K[s_k, DIM]^T   (per request)
  // Column-major encoding of C^T[s_k, H] = K[s_k, DIM] . Q^T[DIM, H].
  mkl::blas::column_major::gemm_batch(
      q, T, N, seqlen, H, DIM,
      scale,
      dkv, DIM,    (std::int64_t)seqlen * DIM,
      dq,  DIM,    (std::int64_t)H * DIM,
      0.f,
      dS,  seqlen, (std::int64_t)H * seqlen,
      B);

  mla_softmax(q, dS, dP, dlse, seqlen, rows);

  // GEMM2: O[H, D_V] = P[H, s_k] . V[s_k, D_V]   (V = first D_V dims of KV)
  // Column-major encoding of C^T[D_V, H] = V^T[D_V, s_k] . P^T[s_k, H].
  mkl::blas::column_major::gemm_batch(
      q, N, N, D_V, H, seqlen,
      1.f,
      dkv,  DIM,    (std::int64_t)seqlen * DIM,
      dP,   seqlen, (std::int64_t)H * seqlen,
      0.f,
      dout, D_V,    (std::int64_t)H * D_V,
      B);
}
