#include <chrono>
#include "helper.h"

// MXFP8 block-scaled matmul in cuBLASLt requires CUDA Toolkit >= 12.8
// (the CUDA_R_8F_E4M3 data type and VEC32_UE8M0 scale mode) and
// compute capability >= 10.0.
#if defined(CUDART_VERSION) && (CUDART_VERSION >= 12080)
#define MXFP8_GEMM_SUPPORTED 1
#else
#define MXFP8_GEMM_SUPPORTED 0
#endif

#if MXFP8_GEMM_SUPPORTED

/// Block-scaled MXFP8 matmul with cublasLtMatmul.
///
/// This mirrors the cuBLAS mxfp8 baseline in reference.py:
///   D(M,N) = (A(M,K) * a_scale) @ (B(N,K) * b_scale)^T
/// A and B are FP8 (CUDA_R_8F_E4M3), one byte per element along K.
/// The block scales are UE8M0 (CUDA_R_8F_UE8M0) with one scale per 32 elements
/// along the K dimension (scale mode CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0).
/// Output D is BF16.
///
/// A is stored (M,K) row-major and B is stored (N,K) row-major. To compute
/// A @ B^T on the column-major cuBLASLt we use transa = OP_T, transb = OP_N so
/// that A is interpreted as (K x M) and B as (K x N), producing D = (M x N).
bool LtMxfp8Matmul(const int repeat,
                   cublasLtHandle_t ltHandle,
                   int m,
                   int n,
                   int k,
                   const float *alpha, /* host pointer */
                   const float *beta,  /* host pointer */
                   const void *a_scale, /* device pointer, UE8M0 block scales */
                   const void *A,       /* device pointer, FP8 E4M3 (M,K) */
                   int lda,
                   const void *b_scale, /* device pointer, UE8M0 block scales */
                   const void *B,       /* device pointer, FP8 E4M3 (N,K) */
                   int ldb,
                   __nv_bfloat16 *D,    /* device pointer, BF16 (M,N) */
                   int ldd,
                   void *workspace,
                   size_t workspaceSize) {
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    // A must be transposed and B non-transposed (TN format required by FP8).
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

    // Create matrix descriptors. FP8 dimensions are in elements (1 byte each).
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, k, m, lda));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, k, n, ldb));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, m, n, ldd));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_16BF, m, n, ldd));

    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0) {
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

#endif // MXFP8_GEMM_SUPPORTED

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <repeat>\n", argv[0]);
        return 1;
    }
    const int repeat = atoi(argv[1]);

#if !MXFP8_GEMM_SUPPORTED
    printf("Skipped: MXFP8 block-scaled matmul requires CUDA Toolkit >= 12.8\n");
    return 0;
#else
    int device = 0;
    checkCudaStatus(cudaGetDevice(&device));
    cudaDeviceProp prop;
    checkCudaStatus(cudaGetDeviceProperties(&prop, device));
    if (prop.major < 10) {
        printf("Skipped: MXFP8 block-scaled matmul requires compute capability >= 10.0, found sm_%d%d (%s)\n",
               prop.major, prop.minor, prop.name);
        return 0;
    }

    const int shapes[6][3] = {{16384, 8192, 1280},
                              {16384, 1024, 8192},
                              {16384, 8192, 7168},
                              {16384, 3584, 8192},
                              {8192, 8192, 8192},
                              {16384, 16384, 16384}};

    for (int i = 0; i < 6; i++) {
        int m = shapes[i][0], n = shapes[i][1], k = shapes[i][2];
        printf("Matrix dimension (M, N, K) = (%d, %d, %d)\n", m, n, k);

        Mxfp8TestBench props(m, n, k, 1.0f, 0.0f, 32ULL * 1024 * 1024);

        bool ran = LtMxfp8Matmul(repeat,
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
#endif // MXFP8_GEMM_SUPPORTED
}
