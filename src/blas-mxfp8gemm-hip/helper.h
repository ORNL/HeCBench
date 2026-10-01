#pragma once

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <vector>

//#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_runtime_api.h>
#include <hipblaslt/hipblaslt.h>

inline void checkHipStatus(hipError_t status) {
    if (status != hipSuccess) {
        printf("HIP API failed with status %d: %s\n", status, hipGetErrorString(status));
        throw std::logic_error("HIP API failed");
    }
}

inline void checkHipblasStatus(hipblasStatus_t status) {
    if (status != HIPBLAS_STATUS_SUCCESS) {
        printf("hipBLAS API failed with status %d\n", status);
        throw std::logic_error("hipBLAS API failed");
    }
}

// Block-scaled MXFP8 GEMM test harness for CDNA4 (gfx950).
//
// Mirrors the mxfp8 microscaling path:
//   D(M,N) = (A(M,K) * a_scale) @ (B(N,K) * b_scale)^T
// A and B are FP8 (E4M3), one byte per element along K (not packed).
// Scales are E8M0 (UE8M0), one scale per 32-element block along K (VEC_SIZE = 32),
// as required by the OCP microscaling (MX) format on CDNA4.
// Output D is BF16.
struct Mxfp8TestBench {
    // VEC_SIZE = 32 for mxfp8 (one UE8M0 scale per 32 FP8 elements along K)
    static constexpr int VEC_SIZE = 32;

    Mxfp8TestBench(int m, int n, int k, float alpha = 1.0f, float beta = 0.0f,
                   size_t workspaceSize = 32ULL * 1024 * 1024)
        : m(m), n(n), k(k), alpha(alpha), beta(beta), workspaceSize(workspaceSize) {
        if (k % VEC_SIZE != 0) throw std::logic_error("K must be a multiple of VEC_SIZE");

        // A: (M, K) FP8 -> M * K bytes; B: (N, K) FP8 -> N * K bytes
        aBytes = (size_t)m * k;
        bBytes = (size_t)n * k;
        // One UE8M0 scale (1 byte) per VEC_SIZE elements along K.
        aScaleBytes = (size_t)m * (k / VEC_SIZE);
        bScaleBytes = (size_t)n * (k / VEC_SIZE);
        dElems = (size_t)m * n;

        checkHipblasStatus(hipblasLtCreate(&ltHandle));
        checkHipStatus(hipMalloc(&Adev, aBytes));
        checkHipStatus(hipMalloc(&Bdev, bBytes));
        checkHipStatus(hipMalloc(&AscaleDev, aScaleBytes));
        checkHipStatus(hipMalloc(&BscaleDev, bScaleBytes));
        checkHipStatus(hipMalloc(reinterpret_cast<void**>(&Ddev), dElems * sizeof(__hip_bfloat16)));
        checkHipStatus(hipMalloc(&workspace, workspaceSize));
        checkHipStatus(hipStreamCreate(&stream));

        fillData();
    }

    ~Mxfp8TestBench() {
        checkHipblasStatus(hipblasLtDestroy(ltHandle));
        checkHipStatus(hipFree(Adev));
        checkHipStatus(hipFree(Bdev));
        checkHipStatus(hipFree(AscaleDev));
        checkHipStatus(hipFree(BscaleDev));
        checkHipStatus(hipFree(Ddev));
        checkHipStatus(hipFree(workspace));
        checkHipStatus(hipStreamDestroy(stream));
    }

    // FP8 E4M3 encoding of 1.0 is 0x38 (exp bias 7 -> exp field 0b0111, mantissa 0).
    static constexpr uint8_t FP8_FILL = 0x38;
    // E8M0 (UE8M0) encoding of 1.0 is 127 (exponent bias 127, so 2^0 = 1).
    static constexpr uint8_t SCALE_FILL = 127;

    void fillData() {
        std::vector<uint8_t> Ah(aBytes, FP8_FILL), Bh(bBytes, FP8_FILL);
        std::vector<uint8_t> Asc(aScaleBytes, SCALE_FILL), Bsc(bScaleBytes, SCALE_FILL);

        checkHipStatus(hipMemcpy(Adev, Ah.data(), aBytes, hipMemcpyHostToDevice));
        checkHipStatus(hipMemcpy(Bdev, Bh.data(), bBytes, hipMemcpyHostToDevice));
        checkHipStatus(hipMemcpy(AscaleDev, Asc.data(), aScaleBytes, hipMemcpyHostToDevice));
        checkHipStatus(hipMemcpy(BscaleDev, Bsc.data(), bScaleBytes, hipMemcpyHostToDevice));
    }

    // Decode an 8-bit E4M3 (FP8) code to float (exponent bias 7).
    static float decodeE4M3(uint8_t code) {
        int sign = (code >> 7) & 0x1;
        int exp  = (code >> 3) & 0xF;
        int man  = code & 0x7;
        float mag;
        if (exp == 0) mag = (man / 8.0f) * ldexpf(1.0f, 1 - 7);      // subnormal
        else          mag = (1.0f + man / 8.0f) * ldexpf(1.0f, exp - 7);
        return sign ? -mag : mag;
    }

    // Decode an 8-bit E8M0 (UE8M0) scale code to float: value = 2^(code - 127).
    static float decodeUE8M0(uint8_t code) {
        return ldexpf(1.0f, (int)code - 127);
    }

    // CPU reference check. The inputs are uniform, so every output element of
    //   D = alpha * (A*a_scale) @ (B*b_scale)^T + beta * C
    // equals alpha * (a_elem*a_scale) * (b_elem*b_scale) * K (with beta = 0).
    bool verify(double relTol = 1e-2) {
        std::vector<__hip_bfloat16> Dh(dElems);
        checkHipStatus(hipMemcpy(Dh.data(), Ddev, dElems * sizeof(__hip_bfloat16), hipMemcpyDeviceToHost));

        float aElem  = decodeE4M3(FP8_FILL);
        float bElem  = decodeE4M3(FP8_FILL);
        float aScale = decodeUE8M0(SCALE_FILL);
        float bScale = decodeUE8M0(SCALE_FILL);
        double expected = (double)alpha * (double)(aElem * aScale) *
                          (double)(bElem * bScale) * (double)k;

        double maxErr = 0.0;
        for (size_t i = 0; i < dElems; i++) {
            double got = (double)__bfloat162float(Dh[i]);
            maxErr = std::max(maxErr, std::fabs(got - expected));
        }
        bool ok = maxErr <= relTol * (std::fabs(expected) + 1.0);
        printf("Verification: expected %.1f, max abs error %.4g -> %s\n",
               expected, maxErr, ok ? "PASS" : "FAIL");
        return ok;
    }

    int m, n, k;
    float alpha, beta;
    size_t workspaceSize;
    size_t aBytes, bBytes, aScaleBytes, bScaleBytes, dElems;

    void *Adev = nullptr, *Bdev = nullptr;           // FP8 (E4M3)
    void *AscaleDev = nullptr, *BscaleDev = nullptr;  // UE8M0 block scales
    __hip_bfloat16 *Ddev = nullptr;                   // BF16 output
    void *workspace = nullptr;
    hipStream_t stream;
    hipblasLtHandle_t ltHandle;
};
