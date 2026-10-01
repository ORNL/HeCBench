#include <chrono>
#include <cstring>
#include "helper.h"

// MXFP6 block-scaled matmul in cuBLASLt requires CUDA Toolkit >= 12.8
// (the CUDA_R_6F_E2M3 data type and VEC32_UE8M0 scale mode) and
// compute capability >= 10.0.
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 12080)
#define MXFP6_GEMM_SUPPORTED 1
#else
#define MXFP6_GEMM_SUPPORTED 0
#endif

#if MXFP6_GEMM_SUPPORTED

/// Block-scaled MXFP6 (FP6 E2M3) matmul with cublasLtMatmul.
///
///   D(M,N) = (A(M,K) * a_scale) @ (B(N,K) * b_scale)^T
/// A and B are FP6 (CUDA_R_6F_E2M3) packed four elements per three bytes along K.
/// The block scales are UE8M0 (CUDA_R_8F_UE8M0) with one scale per 32 elements
/// along the K dimension (scale mode CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0).
/// Output D is BF16.
///
/// A is stored (M,K) row-major and B is stored (N,K) row-major. To compute
/// A @ B^T on the column-major cuBLASLt we use transa = OP_T, transb = OP_N so
/// that A is interpreted as (K x M) and B as (K x N), producing D = (M x N).
/// Returns true if the matmul actually ran (a supported heuristic was found),
/// false if the configuration is unsupported on the current GPU/cuBLASLt.
bool LtMxfp6Matmul(const int repeat,
                   cublasLtHandle_t ltHandle,
                   int m,
                   int n,
                   int k,
                   const float *alpha, /* host pointer */
                   const float *beta,  /* host pointer */
                   const void *a_scale, /* device pointer, UE8M0 block scales */
                   const void *A,       /* device pointer, packed FP6 (M,K) */
                   int lda,
                   const void *b_scale, /* device pointer, UE8M0 block scales */
                   const void *B,       /* device pointer, packed FP6 (N,K) */
                   int ldb,
                   __nv_bfloat16 *D,    /* device pointer, BF16 (M,N) */
                   int ldd,
                   void *workspace,
                   size_t workspaceSize) {
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    // A must be transposed and B non-transposed (TN format required by MX types).
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;

    int returnedResults = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // Block-scaling: set the scale pointers and mark them as VEC32_UE8M0 tensors.
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)));

    cublasLtMatmulMatrixScale_t scaleMode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode)));

    // Create matrix descriptors. FP6 dimensions are in elements (packed 4/3 bytes).
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_6F_E2M3, k, m, lda));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_6F_E2M3, k, n, ldb));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, m, n, ldd));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_16BF, m, n, ldd));

    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    // A NOT_SUPPORTED status here means the current GPU/cuBLASLt lacks an MXFP6
    // block-scaled kernel (it requires datacenter Blackwell, e.g. sm_100/sm_103);
    // treat it like "no heuristic available" and skip gracefully instead of aborting.
    cublasStatus_t heuristicStatus = cublasLtMatmulAlgoGetHeuristic(
        ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1,
        &heuristicResult, &returnedResults);
    if (heuristicStatus != CUBLAS_STATUS_SUCCESS && heuristicStatus != CUBLAS_STATUS_NOT_SUPPORTED) {
        checkCublasStatus(heuristicStatus);
    }

    if (heuristicStatus == CUBLAS_STATUS_NOT_SUPPORTED || returnedResults == 0) {
        printf("no heuristic function available for current configuration\n");
        if (preference) cublasLtMatmulPreferenceDestroy(preference);
        if (Ddesc) cublasLtMatrixLayoutDestroy(Ddesc);
        if (Cdesc) cublasLtMatrixLayoutDestroy(Cdesc);
        if (Bdesc) cublasLtMatrixLayoutDestroy(Bdesc);
        if (Adesc) cublasLtMatrixLayoutDestroy(Adesc);
        if (operationDesc) cublasLtMatmulDescDestroy(operationDesc);
        return false;
    }

    // Warm up
    checkCublasStatus(cublasLtMatmul(ltHandle,
                                     operationDesc,
                                     alpha, A, Adesc,
                                     B, Bdesc, beta,
                                     D, Cdesc,
                                     D, Ddesc,
                                     &heuristicResult.algo,
                                     workspace,
                                     workspaceSize,
                                     0));
    cudaDeviceSynchronize();

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
        checkCublasStatus(cublasLtMatmul(ltHandle,
                                         operationDesc,
                                         alpha, A, Adesc,
                                         B, Bdesc, beta,
                                         D, Cdesc,
                                         D, Ddesc,
                                         &heuristicResult.algo,
                                         workspace,
                                         workspaceSize,
                                         0));
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    auto ns = (time / repeat);
    printf("Average cublasLtMatmul execution time %10.3f (us) | ", ns * 1e-3f);
    printf("Average cublasLtMatmul performance %.1f (TFLOPS)\n", 2.f * m * k * n / ns * 1e-3f);

    if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    if (Ddesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Ddesc));
    if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
    return true;
}

#endif // MXFP6_GEMM_SUPPORTED

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <repeat>\n", argv[0]);
        return 1;
    }
    const int repeat = atoi(argv[1]);

#if !MXFP6_GEMM_SUPPORTED
    printf("Skipped: MXFP6 block-scaled matmul requires CUDA Toolkit >= 12.8\n");
    return 0;
#else
    int device = 0;
    checkCudaStatus(cudaGetDevice(&device));
    cudaDeviceProp prop;
    checkCudaStatus(cudaGetDeviceProperties(&prop, device));
    if (prop.major < 10) {
        printf("Skipped: MXFP6 block-scaled matmul requires compute capability >= 10.0, found sm_%d%d (%s)\n",
               prop.major, prop.minor, prop.name);
        return 0;
    }

    // M = N = 8192, sweep K
    const int M = 8192;
    const int N = 8192;
    const int Ks[5] = {512, 1024, 2048, 4096, 8192};

    for (int i = 0; i < 5; i++) {
        int m = M, n = N, k = Ks[i];
        printf("Matrix dimension (M, N, K) = (%d, %d, %d)\n", m, n, k);

        Mxfp6TestBench props(m, n, k, 1.0f, 0.0f, 32ULL * 1024 * 1024);

        bool ran = LtMxfp6Matmul(repeat,
                      props.ltHandle,
                      props.m,
                      props.n,
                      props.k,
                      &props.alpha,
                      &props.beta,
                      props.AscaleDev, props.Adev, props.k, // A: (M,K), lda = K (elements)
                      props.BscaleDev, props.Bdev, props.k, // B: (N,K), ldb = K (elements)
                      props.Ddev, props.m,                  // D: (M,N), ldd = M
                      props.workspace,
                      props.workspaceSize);

        if (ran) props.verify();
    }

    return 0;
#endif // MXFP6_GEMM_SUPPORTED
}
