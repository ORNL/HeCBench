#include <chrono>
#include <cstring>
#include "helper.h"

// MXFP8 block-scaled matmul in hipBLASLt requires ROCm >= 7.0
#if defined(HIP_VERSION_MAJOR) && (HIP_VERSION_MAJOR >= 7)
#define MXFP8_GEMM_SUPPORTED 1
#else
#define MXFP8_GEMM_SUPPORTED 0
#endif

#if MXFP8_GEMM_SUPPORTED

/// Block-scaled MXFP8 matmul with hipblasLtMatmul on CDNA4 (gfx950).
///
///   D(M,N) = (A(M,K) * a_scale) @ (B(N,K) * b_scale)^T
/// A and B are FP8 (HIP_R_8F_E4M3) one byte per element along K.
/// The block scales are E8M0 (HIP_R_8F_UE8M0) with one scale per 32 elements
/// along the K dimension (scale mode HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0).
/// Output D is BF16.
///
/// A is stored (M,K) row-major and B is stored (N,K) row-major. To compute
/// A @ B^T on the column-major hipBLASLt we use transa = OP_T, transb = OP_N so
/// that A is interpreted as (K x M) and B as (K x N), producing D = (M x N).
bool LtMxfp8Matmul(const int repeat,
                   hipblasLtHandle_t ltHandle,
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
                   __hip_bfloat16 *D,   /* device pointer, BF16 (M,N) */
                   int ldd,
                   void *workspace,
                   size_t workspaceSize) {
    hipblasLtMatmulDesc_t operationDesc = NULL;
    hipblasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;
    hipblasLtMatmulPreference_t preference = NULL;

    // A must be transposed and B non-transposed (TN format required by FP8).
    hipblasOperation_t transa = HIPBLAS_OP_T;
    hipblasOperation_t transb = HIPBLAS_OP_N;

    int returnedResults = 0;
    hipblasLtMatmulHeuristicResult_t heuristicResult = {};

    checkHipblasStatus(hipblasLtMatmulDescCreate(&operationDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    checkHipblasStatus(hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkHipblasStatus(hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // Block-scaling: set the scale pointers and mark them as VEC32_UE8M0 tensors.
    checkHipblasStatus(hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));
    checkHipblasStatus(hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)));

    hipblasLtMatmulMatrixScale_t scaleMode = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    checkHipblasStatus(hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode)));
    checkHipblasStatus(hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode)));

    // Create matrix descriptors. FP8 dimensions are in elements (1 byte each).
    checkHipblasStatus(hipblasLtMatrixLayoutCreate(&Adesc, HIP_R_8F_E4M3, k, m, lda));
    checkHipblasStatus(hipblasLtMatrixLayoutCreate(&Bdesc, HIP_R_8F_E4M3, k, n, ldb));
    checkHipblasStatus(hipblasLtMatrixLayoutCreate(&Cdesc, HIP_R_16BF, m, n, ldd));
    checkHipblasStatus(hipblasLtMatrixLayoutCreate(&Ddesc, HIP_R_16BF, m, n, ldd));

    checkHipblasStatus(hipblasLtMatmulPreferenceCreate(&preference));
    checkHipblasStatus(hipblasLtMatmulPreferenceSetAttribute(preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    checkHipblasStatus(hipblasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0) {
        printf("no heuristic function available for current configuration\n");
        if (preference) hipblasLtMatmulPreferenceDestroy(preference);
        if (Ddesc) hipblasLtMatrixLayoutDestroy(Ddesc);
        if (Cdesc) hipblasLtMatrixLayoutDestroy(Cdesc);
        if (Bdesc) hipblasLtMatrixLayoutDestroy(Bdesc);
        if (Adesc) hipblasLtMatrixLayoutDestroy(Adesc);
        if (operationDesc) hipblasLtMatmulDescDestroy(operationDesc);
        return false;
    }

    // Warm up
    checkHipblasStatus(hipblasLtMatmul(ltHandle,
                                       operationDesc,
                                       alpha, A, Adesc,
                                       B, Bdesc, beta,
                                       D, Cdesc,
                                       D, Ddesc,
                                       &heuristicResult.algo,
                                       workspace,
                                       workspaceSize,
                                       0));
    checkHipStatus(hipDeviceSynchronize());

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
        checkHipblasStatus(hipblasLtMatmul(ltHandle,
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

    checkHipStatus(hipDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    auto ns = (time / repeat);
    printf("Average hipblasLtMatmul execution time %10.3f (us) | ", ns * 1e-3f);
    printf("Average hipblasLtMatmul performance %.1f (TFLOPS)\n", 2.f * m * k * n / ns * 1e-3f);

    if (preference) checkHipblasStatus(hipblasLtMatmulPreferenceDestroy(preference));
    if (Ddesc) checkHipblasStatus(hipblasLtMatrixLayoutDestroy(Ddesc));
    if (Cdesc) checkHipblasStatus(hipblasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkHipblasStatus(hipblasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkHipblasStatus(hipblasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) checkHipblasStatus(hipblasLtMatmulDescDestroy(operationDesc));
    return true;
}

// Runtime capability probe: ask hipBLASLt whether it has any algorithm for a
// block-scaled (VEC32_UE8M0) matmul of the given FP type on the current device.
// This is the authoritative "does this device + library support this FP8 type"
// check: unlike a GPU-architecture string match it needs no white-list and
// automatically covers new hardware (e.g. gfx1250 as well as gfx950). Heuristic
// queries inspect only the descriptors, so we just need small dummy scale
// buffers and no real A/B/C/D data.
static bool deviceSupportsBlockScaled(hipblasLtHandle_t ltHandle, hipDataType abType) {
    const int m = 256, n = 256, k = 256; // any MX-legal shape (k % 128 == 0)
    hipblasLtMatmulDesc_t opDesc = nullptr;
    hipblasLtMatrixLayout_t Ad = nullptr, Bd = nullptr, Cd = nullptr, Dd = nullptr;
    hipblasLtMatmulPreference_t pref = nullptr;
    hipblasOperation_t transa = HIPBLAS_OP_T, transb = HIPBLAS_OP_N;
    hipblasLtMatmulMatrixScale_t scaleMode = HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
    size_t ws = 32ULL * 1024 * 1024;
    int returned = 0;
    hipblasLtMatmulHeuristicResult_t res = {};
    void *aScale = nullptr, *bScale = nullptr;
    bool ok = false;

    if (hipMalloc(&aScale, 256) != hipSuccess) return false;
    if (hipMalloc(&bScale, 256) != hipSuccess) { (void)hipFree(aScale); return false; }

    if (hipblasLtMatmulDescCreate(&opDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F) == HIPBLAS_STATUS_SUCCESS) {
        hipblasLtMatmulDescSetAttribute(opDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
        hipblasLtMatmulDescSetAttribute(opDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
        hipblasLtMatmulDescSetAttribute(opDesc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &aScale, sizeof(aScale));
        hipblasLtMatmulDescSetAttribute(opDesc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &bScale, sizeof(bScale));
        hipblasLtMatmulDescSetAttribute(opDesc, HIPBLASLT_MATMUL_DESC_A_SCALE_MODE, &scaleMode, sizeof(scaleMode));
        hipblasLtMatmulDescSetAttribute(opDesc, HIPBLASLT_MATMUL_DESC_B_SCALE_MODE, &scaleMode, sizeof(scaleMode));
        hipblasLtMatrixLayoutCreate(&Ad, abType, k, m, k);
        hipblasLtMatrixLayoutCreate(&Bd, abType, k, n, k);
        hipblasLtMatrixLayoutCreate(&Cd, HIP_R_16BF, m, n, m);
        hipblasLtMatrixLayoutCreate(&Dd, HIP_R_16BF, m, n, m);
        hipblasLtMatmulPreferenceCreate(&pref);
        hipblasLtMatmulPreferenceSetAttribute(pref, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
        if (hipblasLtMatmulAlgoGetHeuristic(ltHandle, opDesc, Ad, Bd, Cd, Dd, pref, 1, &res, &returned) == HIPBLAS_STATUS_SUCCESS)
            ok = (returned > 0);
    }

    if (pref) hipblasLtMatmulPreferenceDestroy(pref);
    if (Dd) hipblasLtMatrixLayoutDestroy(Dd);
    if (Cd) hipblasLtMatrixLayoutDestroy(Cd);
    if (Bd) hipblasLtMatrixLayoutDestroy(Bd);
    if (Ad) hipblasLtMatrixLayoutDestroy(Ad);
    if (opDesc) hipblasLtMatmulDescDestroy(opDesc);
    (void)hipFree(aScale);
    (void)hipFree(bScale);
    return ok;
}

#endif // MXFP8_GEMM_SUPPORTED

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <repeat>\n", argv[0]);
        return 1;
    }
    const int repeat = atoi(argv[1]);

#if !MXFP8_GEMM_SUPPORTED
    printf("Skipped: MXFP8 block-scaled matmul requires ROCm >= 7.0\n");
    return 0;
#else
    // Support is decided by a runtime capability probe (see above) rather than a
    // hard-coded gfx950 check, so any device whose hipBLASLt provides an MXFP8
    // block-scaled algorithm (gfx950, gfx1250, ...) is allowed to run.
    int device = 0;
    checkHipStatus(hipGetDevice(&device));
    hipDeviceProp_t prop;
    checkHipStatus(hipGetDeviceProperties(&prop, device));

    hipblasLtHandle_t probeHandle;
    checkHipblasStatus(hipblasLtCreate(&probeHandle));
    const bool supported = deviceSupportsBlockScaled(probeHandle, HIP_R_8F_E4M3);
    checkHipblasStatus(hipblasLtDestroy(probeHandle));
    if (!supported) {
        printf("Skipped: MXFP8 block-scaled matmul is not supported on this device "
               "(%s / %s)\n", prop.gcnArchName, prop.name);
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
