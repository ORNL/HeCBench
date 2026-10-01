#include <chrono>
#include "helper.h"

// FP8 encoding differs by GPU architecture:
//   gfx940/gfx942 (MI300)  -> FNUZ  (HIP_R_8F_E4M3_FNUZ)
//   gfx950        (MI350)  -> OCP   (HIP_R_8F_E4M3)
// The __gfx__ preprocessor macros are only defined during device compilation,
// so the selection is done at runtime (host code) by inspecting the arch name.
static hipDataType selectFp8Type() {
    // gfx950 and newer use the OCP FP8 format; gfx94x use FNUZ.
    return fp8IsOcp() ? HIP_R_8F_E4M3 : HIP_R_8F_E4M3_FNUZ;
}

/// Sample wrapper executing fp8 matmul with hipblasLtMatmul, with addition of per-tensor input scaling and the
/// workspace to support split-K algorithms. A and B are FP8; the output D (and C) are BF16.
///
/// pointer mode is for alpha and beta is always host, to change it configure the appropriate matmul descriptor
/// attribute matmul is not using hipblas handle's configuration of math mode, here tensor ops are implicitly allowed; to
/// change this configure appropriate attribute in the preference handle
bool LtFp8Matmul(const int repeat,
                 hipblasLtHandle_t ltHandle,
                 int m,
                 int n,
                 int k,
                 const float *alpha, /* host pointer */
                 const float *beta, /* host pointer */
                 const float *a_scale, /* device pointer */
                 const hipblaslt_f8_fnuz *A,
                 int lda,
                 const float *b_scale, /* device pointer */
                 const hipblaslt_f8_fnuz *B,
                 int ldb,
                 const __hip_bfloat16 *C,
                 int ldc,
                 __hip_bfloat16 *D,
                 void *workspace,
                 size_t workspaceSize,
                 hipDataType fp8Type) {
    hipblasLtMatmulDesc_t operationDesc = NULL;
    hipblasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;
    hipblasLtMatmulPreference_t preference = NULL;

    hipblasOperation_t transa = HIPBLAS_OP_T;
    hipblasOperation_t transb = HIPBLAS_OP_N;

    int returnedResults                             = 0;
    hipblasLtMatmulHeuristicResult_t heuristicResult = {};

    // create operation desciriptor; see hipblasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    checkHipblasStatus(hipblasLtMatmulDescCreate(&operationDesc, HIPBLAS_COMPUTE_32F, HIP_R_32F));
    checkHipblasStatus(hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkHipblasStatus(hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // per-tensor input scaling factors (only A and B are FP8; the BF16 output
    // needs no D scale, and C/D scale + amax are not valid for a BF16 output)
    checkHipblasStatus(hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));
    checkHipblasStatus(hipblasLtMatmulDescSetAttribute(operationDesc, HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)));

    // create matrix descriptors: A and B are FP8, C and D are BF16.
    checkHipblasStatus(hipblasLtMatrixLayoutCreate(&Adesc, fp8Type, transa == HIPBLAS_OP_N ? m : k, transa == HIPBLAS_OP_N ? k : m, lda));
    checkHipblasStatus(hipblasLtMatrixLayoutCreate(&Bdesc, fp8Type, transb == HIPBLAS_OP_N ? k : n, transb == HIPBLAS_OP_N ? n : k, ldb));
    checkHipblasStatus(hipblasLtMatrixLayoutCreate(&Cdesc, HIP_R_16BF, m, n, ldc));
    checkHipblasStatus(hipblasLtMatrixLayoutCreate(&Ddesc, HIP_R_16BF, m, n, ldc));

    // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
    // directly come from hipMalloc)
    checkHipblasStatus(hipblasLtMatmulPreferenceCreate(&preference));
    checkHipblasStatus(hipblasLtMatmulPreferenceSetAttribute(preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
    // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
    checkHipblasStatus(hipblasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0) {
        printf("no heuristic function available for current configuration\n");
        if (operationDesc) checkHipblasStatus(hipblasLtMatmulDescDestroy(operationDesc));
        if (Adesc) checkHipblasStatus(hipblasLtMatrixLayoutDestroy(Adesc));
        if (Bdesc) checkHipblasStatus(hipblasLtMatrixLayoutDestroy(Bdesc));
        if (Cdesc) checkHipblasStatus(hipblasLtMatrixLayoutDestroy(Cdesc));
        if (Ddesc) checkHipblasStatus(hipblasLtMatrixLayoutDestroy(Ddesc));
        if (preference) checkHipblasStatus(hipblasLtMatmulPreferenceDestroy(preference));
        return false;
    }

    // Warm up
    checkHipblasStatus(hipblasLtMatmul(ltHandle,
                                   operationDesc,
                                   alpha, A, Adesc,
                                   B, Bdesc, beta,
                                   C, Cdesc,
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
                                     C, Cdesc,
                                     D, Ddesc,
                                     &heuristicResult.algo,
                                     workspace,
                                     workspaceSize,
                                     0));
    }

    checkHipStatus(hipDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    auto ns_fp8 = (time / repeat);
    printf("Average hipblasLtMatmul execution time %10.3f (us) | ", ns_fp8 * 1e-3f);
    printf("Average hipblasLtMatmul performance %.1f (TFLOPS)\n", 2.f * m * k * n / ns_fp8 * 1e-3f);

    // descriptors are no longer needed as all GPU work was already enqueued
    if (preference) checkHipblasStatus(hipblasLtMatmulPreferenceDestroy(preference));
    if (Ddesc) checkHipblasStatus(hipblasLtMatrixLayoutDestroy(Ddesc));
    if (Cdesc) checkHipblasStatus(hipblasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkHipblasStatus(hipblasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkHipblasStatus(hipblasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) checkHipblasStatus(hipblasLtMatmulDescDestroy(operationDesc));
    return true;
}


int main(int argc, char *argv[])
{
   if (argc != 2) {
     printf("Usage: %s <repeat>\n", argv[0]);
     return 1;
   }
   const int repeat = atoi(argv[1]);

   const hipDataType fp8Type = selectFp8Type();
   printf("Using FP8 E4M3 format: %s\n", fp8Type == HIP_R_8F_E4M3 ? "OCP" : "FNUZ");

   const int shapes[6][3] = {{16384, 8192, 1280},
                             {16384, 1024, 8192},
                             {16384, 8192, 7168},
                             {16384, 3584, 8192},
                             {8192, 8192, 8192},
                             {16384, 16384, 16384}};

   for (int i = 0; i < 6; i++) {

     int m = shapes[i][0], n = shapes[i][1], k = shapes[i][2];
     printf("Matrix dimension (M, N, K) = (%d, %d, %d)\n", m, n, k);

     // A/B are FP8 (hipblaslt_f8_fnuz used as 1-byte storage); C/D are BF16.
     TestBench<hipblaslt_f8_fnuz,
               __hip_bfloat16,
               __hip_bfloat16,
               float> props(m, n, k, 1.0f, 0.0f, 32ULL * 1024 * 1024);

     bool ok = false;
     props.run([&props, repeat, fp8Type, &ok] {
          ok = LtFp8Matmul(repeat,
                      props.ltHandle,
                      props.m,
                      props.n,
                      props.k,
                      &props.alpha,
                      &props.beta,
                      props.AscaleDev, props.Adev, props.k,
                      props.BscaleDev, props.Bdev, props.k,
                      props.Cdev, props.m,
                      props.Ddev,
                      props.workspace,
                      props.workspaceSize,
                      fp8Type);
      });

     if (ok)
       props.verify();
     else
       printf("Skipped: no hipBLASLt kernel available for this GPU and data-type combination\n");
    }

    return 0;
}
