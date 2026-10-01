// hipBLASLt 3-step MLA decode variant:
//     S = scale * Q . K^T      (GEMM1, scale fused into the matmul alpha)
//     P = softmax_row(S)       (dedicated kernel; see note below)
//     O = P . V                (GEMM2)
//
// Portability note: hipBLASLt's solution database on some archs (e.g. gfx90a)
// has no algo for the bf16-in / fp32-out GEMM1. When the heuristic returns no
// algo for a plan we transparently fall back to the legacy
// hipblasGemmStridedBatchedEx (which supports bf16->fp32), preserving the fp32
// score precision the softmax needs. GEMMs that the heuristic can serve still
// run through hipBLASLt.
//
// Requires kernel.h to be included first (reuses mla_softmax_kernel, the CHECK /
// HIPBLAS_CHECK macros, bf16_t, and the DIM/D_V/PAGE/NEG_LARGE macros).

#include <hipblaslt/hipblaslt.h>
#include <algorithm>

// One matmul "plan": a hipBLASLt descriptor/layouts/algo, plus the raw operand
// dims so we can fall back to legacy hipblasGemmStridedBatchedEx when hipBLASLt
// has no algo for the requested type combination.
struct GemmPlan {
  hipblasLtMatmulDesc_t   desc = nullptr;
  hipblasLtMatrixLayout_t A    = nullptr;
  hipblasLtMatrixLayout_t B    = nullptr;
  hipblasLtMatrixLayout_t C    = nullptr;
  hipblasLtMatmulAlgo_t   algo;
  bool                    has_algo = false;

  hipblasOperation_t opA, opB;
  int m, n, k, lda, ldb, ldc, batch;
  long long strideA, strideB, strideC;
  hipDataType Atype, Btype, Ctype;
};

static void destroy_plan(GemmPlan &p) {
  if (p.C)    hipblasLtMatrixLayoutDestroy(p.C);
  if (p.B)    hipblasLtMatrixLayoutDestroy(p.B);
  if (p.A)    hipblasLtMatrixLayoutDestroy(p.A);
  if (p.desc) hipblasLtMatmulDescDestroy(p.desc);
  p = GemmPlan{};
}

// Build a strided-batched matmul plan computing C = op(A) . op(B), where op(A)
// is m x k and op(B) is k x n (column-major, matching hipBLAS legacy semantics).
static void build_plan(GemmPlan &p, hipblasLtHandle_t lt,
                       hipblasOperation_t opA, hipblasOperation_t opB,
                       int m, int n, int k,
                       hipDataType Atype, int lda, long long strideA,
                       hipDataType Btype, int ldb, long long strideB,
                       hipDataType Ctype, int ldc, long long strideC,
                       int batch, size_t wsSize) {
  p.opA = opA; p.opB = opB; p.m = m; p.n = n; p.k = k;
  p.lda = lda; p.ldb = ldb; p.ldc = ldc; p.batch = batch;
  p.strideA = strideA; p.strideB = strideB; p.strideC = strideC;
  p.Atype = Atype; p.Btype = Btype; p.Ctype = Ctype;

  HIPBLAS_CHECK(hipblasLtMatmulDescCreate(&p.desc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
  HIPBLAS_CHECK(hipblasLtMatmulDescSetAttribute(
      p.desc, HIPBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA)));
  HIPBLAS_CHECK(hipblasLtMatmulDescSetAttribute(
      p.desc, HIPBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB)));

  // Layout dims are the *stored* (pre-op) shape: op=N -> as-is, op=T -> swapped.
  int Arows = (opA == HIPBLAS_OP_N) ? m : k;
  int Acols = (opA == HIPBLAS_OP_N) ? k : m;
  int Brows = (opB == HIPBLAS_OP_N) ? k : n;
  int Bcols = (opB == HIPBLAS_OP_N) ? n : k;
  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&p.A, Atype, Arows, Acols, lda));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&p.B, Btype, Brows, Bcols, ldb));
  HIPBLAS_CHECK(hipblasLtMatrixLayoutCreate(&p.C, Ctype, m, n, ldc));

  const int32_t bc = batch;
  auto set_batch = [&](hipblasLtMatrixLayout_t lay, long long stride) {
    HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(
        lay, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &bc, sizeof(bc)));
    int64_t so = stride;
    HIPBLAS_CHECK(hipblasLtMatrixLayoutSetAttribute(
        lay, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &so, sizeof(so)));
  };
  set_batch(p.A, strideA);
  set_batch(p.B, strideB);
  set_batch(p.C, strideC);

  hipblasLtMatmulPreference_t pref = nullptr;
  HIPBLAS_CHECK(hipblasLtMatmulPreferenceCreate(&pref));
  uint64_t ws = wsSize;
  HIPBLAS_CHECK(hipblasLtMatmulPreferenceSetAttribute(
      pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));

  hipblasLtMatmulHeuristicResult_t res{};
  int returned = 0;
  hipblasLtMatmulAlgoGetHeuristic(
      lt, p.desc, p.A, p.B, p.C, p.C, pref, 1, &res, &returned);
  hipblasLtMatmulPreferenceDestroy(pref);
  if (returned > 0) { p.algo = res.algo; p.has_algo = true; }
}

// Run a plan: hipBLASLt if the heuristic served an algo, else legacy hipBLAS.
static void run_plan(hipblasLtHandle_t lt, hipblasHandle_t hb, const GemmPlan &p,
                     const float *alpha, const void *A, const void *B,
                     const float *beta, void *C, void *ws, size_t wsSize) {
  if (p.has_algo) {
    HIPBLAS_CHECK(hipblasLtMatmul(
        lt, p.desc, alpha, A, p.A, B, p.B, beta, C, p.C, C, p.C,
        &p.algo, ws, wsSize, 0));
  } else {
    HIPBLAS_CHECK(hipblasGemmStridedBatchedEx(
        hb, p.opA, p.opB, p.m, p.n, p.k, alpha,
        A, p.Atype, p.lda, (hipblasStride)p.strideA,
        B, p.Btype, p.ldb, (hipblasStride)p.strideB, beta,
        C, p.Ctype, p.ldc, (hipblasStride)p.strideC,
        p.batch, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
  }
}

static void launch_lt(int B, int H, int max_blocks, float scale,
                      const bf16_t *dq, const bf16_t *dkv,
                      bf16_t *dout, float *dlse) {
  const int seqlen = max_blocks * PAGE;

  static float  *dS  = nullptr;
  static bf16_t *dP  = nullptr;
  static size_t  cap = 0;
  const size_t rows = (size_t)B * H;
  const size_t need = rows * seqlen;
  if (need > cap) {
    if (dS) CHECK(hipFree(dS));
    if (dP) CHECK(hipFree(dP));
    CHECK(hipMalloc(&dS, need * sizeof(float)));
    CHECK(hipMalloc(&dP, need * sizeof(bf16_t)));
    cap = need;
  }

  static hipblasLtHandle_t lt = nullptr;
  if (!lt) HIPBLAS_CHECK(hipblasLtCreate(&lt));
  static hipblasHandle_t hb = nullptr;
  if (!hb) HIPBLAS_CHECK(hipblasCreate(&hb));

  static void  *ws     = nullptr;
  static size_t wsSize = 0;
  if (!ws) { wsSize = 32ull * 1024 * 1024; CHECK(hipMalloc(&ws, wsSize)); }

  static GemmPlan g1, g2;
  static int cB = -1, cS = -1;
  if (cB != B || cS != seqlen) {
    destroy_plan(g1);
    destroy_plan(g2);
    // GEMM1: C^T[s_k, H] = K[s_k, DIM] . Q^T[DIM, H]  (scale via alpha)
    build_plan(g1, lt, HIPBLAS_OP_T, HIPBLAS_OP_N, seqlen, H, DIM,
               HIP_R_16BF, DIM,    (long long)seqlen * DIM,
               HIP_R_16BF, DIM,    (long long)H * DIM,
               HIP_R_32F,  seqlen, (long long)H * seqlen,
               B, wsSize);
    // GEMM2: C^T[D_V, H] = V^T[D_V, s_k] . P^T[s_k, H]
    build_plan(g2, lt, HIPBLAS_OP_N, HIPBLAS_OP_N, D_V, H, seqlen,
               HIP_R_16BF, DIM,    (long long)seqlen * DIM,
               HIP_R_16BF, seqlen, (long long)H * seqlen,
               HIP_R_16BF, D_V,    (long long)H * D_V,
               B, wsSize);
    cB = B; cS = seqlen;
  }

  const float alpha = scale, one = 1.f, zero = 0.f;

  run_plan(lt, hb, g1, &alpha, dkv, dq, &zero, dS, ws, wsSize);

  static int max_grid_x = 0;
  if (max_grid_x == 0) {
    int dev = 0;
    CHECK(hipGetDevice(&dev));
    CHECK(hipDeviceGetAttribute(&max_grid_x, hipDeviceAttributeMaxGridDimX, dev));
  }
  const unsigned grid = std::min(rows, (size_t)max_grid_x);
  mla_softmax_kernel<<<grid, 256>>>(dS, dP, dlse, seqlen, rows);

  run_plan(lt, hb, g2, &one, dkv, dP, &zero, dout, ws, wsSize);
}

