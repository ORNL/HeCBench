#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include "helper.h"

// MXFP6 block-scaled matmul in hipBLASLt requires ROCm >= 7.0
#if defined(HIP_VERSION_MAJOR) && (HIP_VERSION_MAJOR >= 7)
#define MXFP6_GEMM_SUPPORTED 1
#else
#define MXFP6_GEMM_SUPPORTED 0
#endif

#if MXFP6_GEMM_SUPPORTED

/// Block-scaled MXFP6 (FP6 E2M3) matmul with hipblasLtMatmul on CDNA4 (gfx950).
///
///   D(M,N) = (A(M,K) * a_scale) @ (B(N,K) * b_scale)^T
/// A and B are FP6 (HIP_R_6F_E2M3) packed four elements per three bytes along K.
/// The block scales are E8M0 (HIP_R_8F_UE8M0) with one scale per 32 elements
/// along the K dimension (scale mode HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0).
/// Output D is BF16.
///
/// A is stored (M,K) row-major and B is stored (N,K) row-major. To compute
/// A @ B^T on the column-major hipBLASLt we use transa = OP_T, transb = OP_N so
/// that A is interpreted as (K x M) and B as (K x N), producing D = (M x N).
bool LtMxfp6Matmul(const int repeat,
                   hipblasLtHandle_t ltHandle,
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
                   __hip_bfloat16 *D,   /* device pointer, BF16 (M,N) */
                   int ldd,
                   void *workspace,
                   size_t workspaceSize) {
    hipblasLtMatmulDesc_t operationDesc = NULL;
    hipblasLtMatrixLayout_t Adesc = NULL, Bdesc = NULL, Cdesc = NULL, Ddesc = NULL;
    hipblasLtMatmulPreference_t preference = NULL;

    // A must be transposed and B non-transposed (TN format required by MX types).
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

    // Create matrix descriptors. FP6 dimensions are in elements (packed 4/3 bytes).
    checkHipblasStatus(hipblasLtMatrixLayoutCreate(&Adesc, HIP_R_6F_E2M3, k, m, lda));
    checkHipblasStatus(hipblasLtMatrixLayoutCreate(&Bdesc, HIP_R_6F_E2M3, k, n, ldb));
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
// This is the authoritative "does this device + library support this FP type"
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

// aiter f6gemm (gfx950). D = alpha * A @ B^T, with A and B in the
// mxfp6_c0c1_256_padk2 layout. The kernel writes row-major BF16; this path
// transposes that into the column-major D buffer the shared verify() reads.
// Packing is outside the timed region, matching hipBLASLt (its inputs are
// already on device when hipblasLtMatmul is timed).
struct __attribute__((packed, aligned(16))) AiterKernelArgs {
    void *ptr_D; unsigned _p0[2];
    void *ptr_C; unsigned _p1[2];
    void *ptr_A; unsigned _p2[2];
    void *ptr_B; unsigned _p3[2];
    float alpha; unsigned _p4[3];
    float beta; unsigned _p5[3];
    unsigned stride_D0; unsigned _p6[3];
    unsigned stride_D1; unsigned _p7[3];
    unsigned stride_C0; unsigned _p8[3];
    unsigned stride_C1; unsigned _p9[3];
    unsigned stride_A0; unsigned _p10[3];
    unsigned stride_A1; unsigned _p11[3];
    unsigned stride_B0; unsigned _p12[3];
    unsigned stride_B1; unsigned _p13[3];
    unsigned M; unsigned _p14[3];
    unsigned N; unsigned _p15[3];
    unsigned K; unsigned _p16[3];
    void *ptr_ScaleA; unsigned _p17[2];
    void *ptr_ScaleB; unsigned _p18[2];
    unsigned stride_ScaleA0; unsigned _p19[3];
    unsigned stride_ScaleA1; unsigned _p20[3];
    unsigned stride_ScaleB0; unsigned _p21[3];
    unsigned stride_ScaleB1; unsigned _p22[3];
    int log2_k_split; unsigned _p23[3];
};
static_assert(sizeof(AiterKernelArgs) == 0x180, "aiter f6gemm kernarg must be 0x180 bytes");
static_assert(offsetof(AiterKernelArgs, ptr_A) == 0x20, "ptr_A");
static_assert(offsetof(AiterKernelArgs, alpha) == 0x40, "alpha");
static_assert(offsetof(AiterKernelArgs, stride_D0) == 0x60, "stride_D0");
static_assert(offsetof(AiterKernelArgs, M) == 0xe0, "M");
static_assert(offsetof(AiterKernelArgs, ptr_ScaleA) == 0x110, "ptr_ScaleA");
static_assert(offsetof(AiterKernelArgs, stride_ScaleA0) == 0x130, "stride_ScaleA0");
static_assert(offsetof(AiterKernelArgs, stride_ScaleB0) == 0x150, "stride_ScaleB0");

static constexpr int kAiterTile = 256;
static constexpr int kAiterKTile = 128;
static constexpr int kAiterPadK = 2;
static constexpr int kAiterPackedTileBytes = 24576;
static constexpr int kAiterScaleTileBytes = 1024;
// One 32-element group per scale byte. A 256x128 tile holds 1024 of them,
// which is also the scale tile size; keep the two counts tied together.
static constexpr int kAiterGroupsPerTile = (kAiterTile * kAiterKTile) / Mxfp6TestBench::VEC_SIZE;
static_assert(kAiterGroupsPerTile == kAiterScaleTileBytes, "scale tile holds one byte per group");
static constexpr const char *kAiterKernelName = "_ZN5aiter29f6gemm_bf16_per1x32Fp6_dmabigE";
// Same relative path AiterAsmKernel loads: $AITER_ASM_DIR/<arch>/f6gemm/<file>.
static constexpr const char *kAiterCoRel = "f6gemm/f6gemm_bf16_per1x32Fp6_dmabig.co";

__global__ void mxfp6RowMajorToColumnMajor(const __hip_bfloat16 *__restrict__ src,
                                           __hip_bfloat16 *__restrict__ dst,
                                           int m, int n) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < m && j < n)
        dst[(size_t)j * m + i] = src[(size_t)i * n + j];
}

static void pack32(const uint8_t codes[32], uint8_t out[24]) {
    for (int g = 0; g < 8; ++g) {
        const unsigned c0 = codes[g * 4 + 0] & 0x3f;
        const unsigned c1 = codes[g * 4 + 1] & 0x3f;
        const unsigned c2 = codes[g * 4 + 2] & 0x3f;
        const unsigned c3 = codes[g * 4 + 3] & 0x3f;
        out[g * 3 + 0] = (uint8_t)((c0 | (c1 << 6)) & 0xff);
        out[g * 3 + 1] = (uint8_t)(((c1 >> 2) | (c2 << 4)) & 0xff);
        out[g * 3 + 2] = (uint8_t)(((c2 >> 4) | (c3 << 2)) & 0xff);
    }
}

// Host port of aiter pack_big / pack_scale. codes is [rows, K] row-major
// 6-bit values; scales is [rows, K/32] row-major UE8M0 codes.
static void packAiterOperand(const std::vector<uint8_t> &codes, int rows, int k,
                             std::vector<uint8_t> &packed) {
    const int nt = rows / kAiterTile;
    const int nk = k / kAiterKTile;
    const int nkPad = nk + kAiterPadK;
    packed.assign((size_t)nt * nkPad * kAiterPackedTileBytes, 0);
    for (int t = 0; t < nt; ++t) {
        for (int s = 0; s < nk; ++s) {
            uint8_t *tile = packed.data() + ((size_t)t * nkPad + s) * kAiterPackedTileBytes;
            for (int i = 0; i < kAiterGroupsPerTile; ++i) {
                const int rb = i / 64;
                const int L = i % 64;
                const int row = t * kAiterTile + rb * 16 + (L % 16);
                const int col = s * kAiterKTile + (L / 16) * Mxfp6TestBench::VEC_SIZE;
                uint8_t block[32];
                for (int e = 0; e < 32; ++e)
                    block[e] = codes[(size_t)row * k + col + e];
                uint8_t bytes[24];
                pack32(block, bytes);
                std::memcpy(tile + (size_t)i * 16, bytes, 16);
                std::memcpy(tile + 16384 + (size_t)i * 8, bytes + 16, 8);
            }
        }
    }
}

static void packAiterScales(const std::vector<uint8_t> &scales, int rows, int k,
                            std::vector<uint8_t> &packed) {
    const int numBlocks = k / Mxfp6TestBench::VEC_SIZE;
    const int nt = rows / kAiterTile;
    const int nk = numBlocks / 4;
    const int nkPad = nk + kAiterPadK;
    packed.assign((size_t)nt * nkPad * kAiterScaleTileBytes, 127);
    for (int t = 0; t < nt; ++t) {
        for (int s = 0; s < nk; ++s) {
            uint8_t *tile = packed.data() + ((size_t)t * nkPad + s) * kAiterScaleTileBytes;
            for (int i = 0; i < kAiterScaleTileBytes; ++i) {
                const int su = i / 512;
                const int within = i % 512;
                const int kg = within / 128;
                const int r16 = (i % 128) / 8;
                const int sub = i % 8;
                const int row = t * kAiterTile + su * 128 + sub * 16 + r16;
                const int block = s * 4 + kg;
                tile[i] = scales[(size_t)row * numBlocks + block];
            }
        }
    }
}

// AITER_ASM_DIR is the hsa directory of an aiter checkout. `import aiter`
// sets it to <aiter>/hsa (develop install) or <aiter>/aiter_meta/hsa (wheel).
// The kernel file is $AITER_ASM_DIR/<arch>/f6gemm/f6gemm_bf16_per1x32Fp6_dmabig.co.
static void printAiterAsmDirHelp(const char *arch) {
    printf("Set AITER_ASM_DIR to the hsa directory of an aiter checkout:\n"
           "  export AITER_ASM_DIR=/path/to/aiter/hsa\n"
           "The %s kernel is then $AITER_ASM_DIR/%s/%s\n",
           arch, arch, kAiterCoRel);
}

static std::string aiterCodeObjectPath(const std::string &arch) {
    const char *asmDir = std::getenv("AITER_ASM_DIR");
    if (asmDir == nullptr || asmDir[0] == '\0') return {};
    std::string dir(asmDir);
    while (!dir.empty() && dir.back() == '/') dir.pop_back();
    return dir + "/" + arch + "/" + kAiterCoRel;
}

static bool loadAiterKernel(const std::string &arch, hipFunction_t *fn) {
    static hipModule_t module = nullptr;
    static hipFunction_t function = nullptr;
    static std::string loadedArch;
    if (function && loadedArch == arch) {
        *fn = function;
        return true;
    }
    const char *asmDir = std::getenv("AITER_ASM_DIR");
    if (asmDir == nullptr || asmDir[0] == '\0') {
        printf("AITER_ASM_DIR is not set.\n");
        printAiterAsmDirHelp(arch.c_str());
        return false;
    }
    const std::string path = aiterCodeObjectPath(arch);
    std::ifstream probe(path);
    if (!probe) {
        printf("aiter code object not found: %s\n", path.c_str());
        printAiterAsmDirHelp(arch.c_str());
        return false;
    }
    checkHipStatus(hipModuleLoad(&module, path.c_str()));
    checkHipStatus(hipModuleGetFunction(&function, module, kAiterKernelName));
    loadedArch = arch;
    *fn = function;
    printf("aiter code object: %s\n", path.c_str());
    return true;
}

bool AiterMxfp6Matmul(const int repeat, Mxfp6TestBench &props) {
    const int m = props.m, n = props.n, k = props.k;
    if (m % kAiterTile || n % kAiterTile || k % kAiterKTile) {
        printf("aiter mxfp6 gemm needs M and N multiples of %d and K a multiple of %d\n",
               kAiterTile, kAiterKTile);
        return false;
    }
    if (props.beta != 0.0f) {
        printf("aiter mxfp6 gemm supports only beta = 0\n");
        return false;
    }
    // dmabig swizzle bounds apply only for K <= 6144. These benchmark shapes fit;
    // refuse anything that would need the swizzle-free fallback kernel.
    if (k <= 6144 && (m > 131072 || n > 16384)) {
        printf("aiter dmabig kernel swizzle bounds exceeded for (M, N, K) = (%d, %d, %d)\n",
               m, n, k);
        return false;
    }

    int device = 0;
    checkHipStatus(hipGetDevice(&device));
    hipDeviceProp_t prop;
    checkHipStatus(hipGetDeviceProperties(&prop, device));
    std::string arch = prop.gcnArchName;
    const size_t colon = arch.find(':');
    if (colon != std::string::npos) arch.resize(colon);

    hipFunction_t fn = nullptr;
    if (!loadAiterKernel(arch, &fn)) return false;

    std::vector<uint8_t> aPacked, bPacked, aScale, bScale;
    packAiterOperand(props.Acodes, m, k, aPacked);
    packAiterOperand(props.Bcodes, n, k, bPacked);
    packAiterScales(props.Ascales, m, k, aScale);
    packAiterScales(props.Bscales, n, k, bScale);

    // Freed on every exit, including a throw from checkHipStatus.
    struct DevFree {
        void *p = nullptr;
        ~DevFree() { if (p) (void)hipFree(p); }
    } aDev, bDev, asDev, bsDev, rowDev;
    auto alloc = [](size_t bytes) {
        void *p = nullptr;
        checkHipStatus(hipMalloc(&p, bytes));
        return p;
    };
    aDev.p = alloc(aPacked.size());
    bDev.p = alloc(bPacked.size());
    asDev.p = alloc(aScale.size());
    bsDev.p = alloc(bScale.size());
    rowDev.p = alloc(props.dElems * sizeof(__hip_bfloat16));
    checkHipStatus(hipMemcpy(aDev.p, aPacked.data(), aPacked.size(), hipMemcpyHostToDevice));
    checkHipStatus(hipMemcpy(bDev.p, bPacked.data(), bPacked.size(), hipMemcpyHostToDevice));
    checkHipStatus(hipMemcpy(asDev.p, aScale.data(), aScale.size(), hipMemcpyHostToDevice));
    checkHipStatus(hipMemcpy(bsDev.p, bScale.data(), bScale.size(), hipMemcpyHostToDevice));

    AiterKernelArgs args{};
    args.ptr_D = rowDev.p;
    args.ptr_A = aDev.p;
    args.ptr_B = bDev.p;
    args.alpha = props.alpha;
    args.stride_D0 = (unsigned)n;
    args.M = (unsigned)m;
    args.N = (unsigned)n;
    args.K = (unsigned)k;
    args.ptr_ScaleA = asDev.p;
    args.ptr_ScaleB = bsDev.p;
    args.stride_ScaleA0 = (unsigned)(k / Mxfp6TestBench::VEC_SIZE);
    args.stride_ScaleB0 = (unsigned)(k / Mxfp6TestBench::VEC_SIZE);

    size_t argSize = sizeof(args);
    void *config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
                      HIP_LAUNCH_PARAM_BUFFER_SIZE, &argSize,
                      HIP_LAUNCH_PARAM_END};
    const unsigned gdx = (unsigned)(n / kAiterTile);
    const unsigned gdy = (unsigned)(m / kAiterTile);
    auto launch = [&]() {
        checkHipStatus(hipModuleLaunchKernel(fn, gdx, gdy, 1, 256, 1, 1, 0, 0, nullptr, config));
    };

    launch();
    checkHipStatus(hipDeviceSynchronize());

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++) launch();
    checkHipStatus(hipDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    auto ns = (time / repeat);
    printf("Average aiter mxfp6 gemm execution time %10.3f (us) | ", ns * 1e-3f);
    printf("Average aiter mxfp6 gemm performance %.1f (TFLOPS)\n", 2.f * m * k * n / ns * 1e-3f);

    dim3 block(16, 16);
    dim3 grid((n + block.x - 1) / block.x, (m + block.y - 1) / block.y);
    mxfp6RowMajorToColumnMajor<<<grid, block>>>(
        static_cast<__hip_bfloat16 *>(rowDev.p), props.Ddev, m, n);
    checkHipStatus(hipGetLastError());
    checkHipStatus(hipDeviceSynchronize());
    return true;
}

#endif // MXFP6_GEMM_SUPPORTED

int main(int argc, char *argv[]) {
    if (argc != 2 && argc != 3) {
        printf("Usage: %s <repeat> [hipblaslt|aiter]\n", argv[0]);
        printf("  hipblaslt is the default.\n");
        printf("  aiter: export AITER_ASM_DIR=/path/to/aiter/hsa\n");
        return 1;
    }
    const int repeat = atoi(argv[1]);
    if (repeat < 1) {
        printf("Usage: %s <repeat> [hipblaslt|aiter]\n", argv[0]);
        printf("  <repeat> must be a positive integer.\n");
        return 1;
    }
    const bool useAiter = (argc == 3);
    if (useAiter && std::strcmp(argv[2], "aiter") != 0 && std::strcmp(argv[2], "hipblaslt") != 0) {
        printf("Usage: %s <repeat> [hipblaslt|aiter]\n", argv[0]);
        printf("  hipblaslt is the default.\n");
        printf("  aiter: export AITER_ASM_DIR=/path/to/aiter/hsa\n");
        return 1;
    }
    const bool aiterBackend = useAiter && std::strcmp(argv[2], "aiter") == 0;

#if !MXFP6_GEMM_SUPPORTED
    printf("Skipped: MXFP6 block-scaled matmul requires ROCm >= 7.0\n");
    return 0;
#else
    // Support is decided by a runtime capability probe (see above) rather than a
    // hard-coded gfx950 check, so any device whose hipBLASLt provides an MXFP6
    // block-scaled algorithm (gfx950, gfx1250, ...) is allowed to run.
    int device = 0;
    checkHipStatus(hipGetDevice(&device));
    hipDeviceProp_t prop;
    checkHipStatus(hipGetDeviceProperties(&prop, device));

    hipblasLtHandle_t probeHandle;
    checkHipblasStatus(hipblasLtCreate(&probeHandle));
    const bool supported = deviceSupportsBlockScaled(probeHandle, HIP_R_6F_E2M3);
    checkHipblasStatus(hipblasLtDestroy(probeHandle));
    // hipBLASLt's heuristic is the gate for the default backend. An explicit
    // aiter run does not depend on that algorithm; it loads its own code object.
    if (!supported && !aiterBackend) {
        printf("Skipped: MXFP6 block-scaled matmul is not supported on this device "
               "(%s / %s)\n", prop.gcnArchName, prop.name);
        return 0;
    }

    if (aiterBackend) {
        std::string arch = prop.gcnArchName;
        const size_t colon = arch.find(':');
        if (colon != std::string::npos) arch.resize(colon);
        hipFunction_t fn = nullptr;
        if (!loadAiterKernel(arch, &fn)) return 1;
    }

    // M = N = 8192, sweep K -- following the benchmark shapes in reference.py.
    const int M = 8192;
    const int N = 8192;
    const int Ks[5] = {512, 1024, 2048, 4096, 8192};

    int failures = 0;
    for (int i = 0; i < 5; i++) {
        int m = M, n = N, k = Ks[i];
        printf("Matrix dimension (M, N, K) = (%d, %d, %d)\n", m, n, k);

        Mxfp6TestBench props(m, n, k, 1.0f, 0.0f, 32ULL * 1024 * 1024);

        bool ran;
        if (aiterBackend) {
            ran = AiterMxfp6Matmul(repeat, props);
        } else {
            ran = LtMxfp6Matmul(repeat,
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
        }

        if (!ran || !props.verify()) failures++;
    }

    return failures == 0 ? 0 : 1;
#endif // MXFP6_GEMM_SUPPORTED
}
