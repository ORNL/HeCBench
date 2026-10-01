// cuBLASLt 3-step MLA decode variant:
//     S = scale * Q . K^T      (GEMM1, scale fused into the matmul alpha)
//     P = softmax_row(S)       (dedicated kernel; see note below)
//     O = P . V                (GEMM2)
//
// This mirrors kernel.h's cuBLAS path but drives the two batched GEMMs through
// the cuBLASLt matmul API (descriptor + cached heuristic algo + a
// workspace). cuBLASLt has NO softmax epilogue and the row-softmax is a genuine
// cross-column reduction (not a pointwise op), so it cannot be fused into a
// matmul epilogue -- the softmax stays a dedicated kernel. What IS fused here is
// the QK scale, folded into GEMM1's alpha. Real softmax<->GEMM fusion needs the
// hand-written flash-attention path (kernel1_wgmma.cuh), not a BLAS library.
//
// Requires kernel.h to be included first (reuses mla_softmax_kernel, the CHECK /
// CUBLAS_CHECK macros, bf16_t, and the DIM/D_V/PAGE/NEG_LARGE macros).

#include <cublasLt.h>
#include <algorithm>

// One cuBLASLt matmul "plan": descriptor + operand layouts + a heuristic algo.
// Cached and rebuilt only when the (batch, seqlen) shape changes.
struct GemmPlan {
  cublasLtMatmulDesc_t   desc = nullptr;
  cublasLtMatrixLayout_t A    = nullptr;
  cublasLtMatrixLayout_t B    = nullptr;
  cublasLtMatrixLayout_t C    = nullptr;
  cublasLtMatmulAlgo_t   algo;
  bool                   has_algo = false;
};

static void destroy_plan(GemmPlan &p) {
  if (p.C)    cublasLtMatrixLayoutDestroy(p.C);
  if (p.B)    cublasLtMatrixLayoutDestroy(p.B);
  if (p.A)    cublasLtMatrixLayoutDestroy(p.A);
  if (p.desc) cublasLtMatmulDescDestroy(p.desc);
  p = GemmPlan{};
}

// Build a strided-batched matmul plan computing C = op(A) . op(B), where op(A)
// is m x k and op(B) is k x n (column-major, matching cuBLAS legacy semantics).
static void build_plan(GemmPlan &p, cublasLtHandle_t lt,
                       cublasOperation_t opA, cublasOperation_t opB,
                       int m, int n, int k,
                       cudaDataType Atype, int lda, long long strideA,
                       cudaDataType Btype, int ldb, long long strideB,
                       cudaDataType Ctype, int ldc, long long strideC,
                       int batch, size_t wsSize) {
  CUBLAS_CHECK(cublasLtMatmulDescCreate(&p.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
  CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
      p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

  // Layout dims are the *stored* (pre-op) shape: op=N -> as-is, op=T -> swapped.
  int Arows = (opA == CUBLAS_OP_N) ? m : k;
  int Acols = (opA == CUBLAS_OP_N) ? k : m;
  int Brows = (opB == CUBLAS_OP_N) ? k : n;
  int Bcols = (opB == CUBLAS_OP_N) ? n : k;
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.A, Atype, Arows, Acols, lda));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.B, Btype, Brows, Bcols, ldb));
  CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&p.C, Ctype, m, n, ldc));

  const int32_t bc = batch;
  auto set_batch = [&](cublasLtMatrixLayout_t lay, long long stride) {
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        lay, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)));
    int64_t so = stride;
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        lay, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &so, sizeof(so)));
  };
  set_batch(p.A, strideA);
  set_batch(p.B, strideB);
  set_batch(p.C, strideC);

  cublasLtMatmulPreference_t pref = nullptr;
  CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
  uint64_t ws = wsSize;
  CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
      pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));

  cublasLtMatmulHeuristicResult_t res{};
  int returned = 0;
  CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
      lt, p.desc, p.A, p.B, p.C, p.C, pref, 1, &res, &returned));
  cublasLtMatmulPreferenceDestroy(pref);
  if (returned > 0) { p.algo = res.algo; p.has_algo = true; }
}

static void launch_lt(int B, int H, int max_blocks, float scale,
                      const bf16_t *dq, const bf16_t *dkv,
                      bf16_t *dout, float *dlse) {
  const int seqlen = max_blocks * PAGE;

  // Scratch: scores S (fp32) and probabilities P (bf16), [B, H, seqlen].
  static float  *dS  = nullptr;
  static bf16_t *dP  = nullptr;
  static size_t  cap = 0;
  const size_t rows = (size_t)B * H;
  const size_t need = rows * seqlen;
  if (need > cap) {
    if (dS) CHECK(cudaFree(dS));
    if (dP) CHECK(cudaFree(dP));
    CHECK(cudaMalloc(&dS, need * sizeof(float)));
    CHECK(cudaMalloc(&dP, need * sizeof(bf16_t)));
    cap = need;
  }

  static cublasLtHandle_t lt = nullptr;
  if (!lt) CUBLAS_CHECK(cublasLtCreate(&lt));

  static void  *ws     = nullptr;
  static size_t wsSize = 0;
  if (!ws) { wsSize = 32ull * 1024 * 1024; CHECK(cudaMalloc(&ws, wsSize)); }

  // Cached plans; rebuilt only when the shape changes.
  static GemmPlan g1, g2;
  static int cB = -1, cS = -1;
  if (cB != B || cS != seqlen) {
    destroy_plan(g1);
    destroy_plan(g2);
    // GEMM1: C^T[s_k, H] = K[s_k, DIM] . Q^T[DIM, H]  (scale via alpha)
    build_plan(g1, lt, CUBLAS_OP_T, CUBLAS_OP_N, seqlen, H, DIM,
               CUDA_R_16BF, DIM,    (long long)seqlen * DIM,
               CUDA_R_16BF, DIM,    (long long)H * DIM,
               CUDA_R_32F,  seqlen, (long long)H * seqlen,
               B, wsSize);
    // GEMM2: C^T[D_V, H] = V^T[D_V, s_k] . P^T[s_k, H]
    build_plan(g2, lt, CUBLAS_OP_N, CUBLAS_OP_N, D_V, H, seqlen,
               CUDA_R_16BF, DIM,    (long long)seqlen * DIM,
               CUDA_R_16BF, seqlen, (long long)H * seqlen,
               CUDA_R_16BF, D_V,    (long long)H * D_V,
               B, wsSize);
    cB = B; cS = seqlen;
  }

  const float alpha = scale, one = 1.f, zero = 0.f;

  CUBLAS_CHECK(cublasLtMatmul(
      lt, g1.desc, &alpha,
      dkv, g1.A, dq, g1.B,
      &zero, dS, g1.C, dS, g1.C,
      g1.has_algo ? &g1.algo : nullptr, ws, wsSize, 0));

  static int max_grid_x = 0;
  if (max_grid_x == 0) {
    int dev = 0;
    CHECK(cudaGetDevice(&dev));
    CHECK(cudaDeviceGetAttribute(&max_grid_x, cudaDevAttrMaxGridDimX, dev));
  }
  const unsigned grid = std::min(rows, (size_t)max_grid_x);
  mla_softmax_kernel<<<grid, 256>>>(dS, dP, dlse, seqlen, rows);

  CUBLAS_CHECK(cublasLtMatmul(
      lt, g2.desc, &one,
      dkv, g2.A, dP, g2.B,
      &zero, dout, g2.C, dout, g2.C,
      g2.has_algo ? &g2.algo : nullptr, ws, wsSize, 0));
}
