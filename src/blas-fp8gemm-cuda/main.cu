#include <chrono>
#include "helper.h"

/// Sample wrapper executing fp8 matmul with cublasLtMatmul, with addition of per-tensor input scaling and the
/// workspace to support split-K algorithms. A and B are FP8; the output D (and C) are BF16.
///
/// pointer mode is for alpha and beta is always host, to change it configure the appropriate matmul descriptor
/// attribute matmul is not using cublas handle's configuration of math mode, here tensor ops are implicitly allowed; to
/// change this configure appropriate attribute in the preference handle
bool LtFp8Matmul(const int repeat,
                 cublasLtHandle_t ltHandle,
                 int m,
                 int n,
                 int k,
                 const float *alpha, /* host pointer */
                 const float *beta, /* host pointer */
                 const float *a_scale, /* device pointer */
                 const __nv_fp8_e4m3 *A,
                 int lda,
                 const float *b_scale, /* device pointer */
                 const __nv_fp8_e4m3 *B,
                 int ldb,
                 const __nv_bfloat16 *C,
                 int ldc,
                 __nv_bfloat16 *D,
                 void *workspace,
                 size_t workspaceSize) {
    cublasLtMatmulDesc_t operationDesc = NULL;
    cublasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;
    cublasLtMatmulPreference_t preference = NULL;

    // A must be transposed and B non-transposed
    cublasOperation_t transa = CUBLAS_OP_T;
    cublasOperation_t transb = CUBLAS_OP_N;

    int returnedResults                             = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    // create operation desciriptor; see cublasLtMatmulDescAttributes_t for details about defaults; here we just need to
    // set the transforms for A and B
    checkCublasStatus(cublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb)));

    // per-tensor input scaling factors (only A and B are FP8; the BF16 output
    // needs no D scale, and C/D scale + amax are not valid for a BF16 output)
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale, sizeof(a_scale)));
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale, sizeof(b_scale)));

    // set fast accumulation
    int8_t fast_accum = 1;
    checkCublasStatus(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_FAST_ACCUM, &fast_accum, sizeof(fast_accum)));

    // create matrix descriptors: A and B are FP8, C and D are BF16.
    // table of supported type combinations can be found in the documentation: https://docs.nvidia.com/cuda/cublas/index.html#cublasltmatmul
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Adesc, CUDA_R_8F_E4M3, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Bdesc, CUDA_R_8F_E4M3, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_16BF, m, n, ldc));
    checkCublasStatus(cublasLtMatrixLayoutCreate(&Ddesc, CUDA_R_16BF, m, n, ldc));

    // create preference handle; here we could use extra attributes to disable tensor ops or to make sure algo selected
    // will work with badly aligned A, B, C; here for simplicity we just assume A,B,C are always well aligned (e.g.
    // directly come from cudaMalloc)
    checkCublasStatus(cublasLtMatmulPreferenceCreate(&preference));
    checkCublasStatus(cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    // we just need the best available heuristic to try and run matmul. There is no guarantee this will work, e.g. if A
    // is badly aligned, you can request more (e.g. 32) algos and try to run them one by one until something works
    checkCublasStatus(cublasLtMatmulAlgoGetHeuristic(ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Ddesc, preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0) {
        printf("no heuristic function available for current configuration\n");
        if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
        if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
        if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
        if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
        if (Ddesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Ddesc));
        if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
        return false;
    }

    // Warm up
    checkCublasStatus(cublasLtMatmul(ltHandle,
                                   operationDesc,
                                   alpha, A, Adesc,
                                   B, Bdesc, beta,
                                   C, Cdesc,
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
                                     C, Cdesc,
                                     D, Ddesc,
                                     &heuristicResult.algo,
                                     workspace,
                                     workspaceSize,
                                     0));
    }

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    auto ns_fp8 = (time / repeat);
    printf("Average cublasLtMatmul execution time %10.3f (us) | ", ns_fp8 * 1e-3f);
    printf("Average cublasLtMatmul performance %.1f (TFLOPS)\n", 2.f * m * k * n / ns_fp8 * 1e-3f);

    // descriptors are no longer needed as all GPU work was already enqueued
    if (preference) checkCublasStatus(cublasLtMatmulPreferenceDestroy(preference));
    if (Ddesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Ddesc));
    if (Cdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Cdesc));
    if (Bdesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Bdesc));
    if (Adesc) checkCublasStatus(cublasLtMatrixLayoutDestroy(Adesc));
    if (operationDesc) checkCublasStatus(cublasLtMatmulDescDestroy(operationDesc));
    return true;
}


int main(int argc, char *argv[])
{
   if (argc != 2) {
     printf("Usage: %s <repeat>\n", argv[0]);
     return 1;
   }
   const int repeat = atoi(argv[1]);

   // FP8 (E4M3) tensor-core matmul in cuBLASLt requires compute capability >= 9.0 (Hopper).
   int device = 0;
   checkCudaStatus(cudaGetDevice(&device));
   cudaDeviceProp prop;
   checkCudaStatus(cudaGetDeviceProperties(&prop, device));
   if (prop.major < 9) {
     printf("Skipped: FP8 matmul requires compute capability >= 9.0, found sm_%d%d (%s)\n",
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

     // A/B are FP8; C/D are BF16.
     TestBench<__nv_fp8_e4m3,  // A/B input type
               __nv_bfloat16,  // C type
               __nv_bfloat16,  // output type
               float> props(m, n, k, 1.0f, 0.0f, 32ULL * 1024 * 1024);

     bool ok = false;
     props.run([&props, repeat, &ok] {
          ok = LtFp8Matmul(repeat,
                      props.ltHandle,
                      props.m,
                      props.n,
                      props.k,
                      &props.alpha,
                      &props.beta,
                      props.AscaleDev, props.Adev, props.k, // mxk
                      props.BscaleDev, props.Bdev, props.k, // kxn
                      props.Cdev, props.m, // mxn
                      props.Ddev,
                      props.workspace,
                      props.workspaceSize);
      });

     if (ok)
       props.verify();
     else
       printf("Skipped: no cuBLASLt kernel available for this GPU and data-type combination\n");
    }

    return 0;
}
