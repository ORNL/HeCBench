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

// Block-scaled FP4 (mxfp4) GEMM test harness for CDNA4 (gfx950).
//
// Mirrors the AMD mxfp4 path of reference.py:
//   D(M,N) = (A(M,K) * a_scale) @ (B(N,K) * b_scale)^T
// A and B are FP4 (E2M1) packed two elements per byte along K.
// Scales are E8M0 (UE8M0), one scale per 32-element block along K (VEC_SIZE = 32),
// as required by the MFMA scaled instructions on CDNA4.
// Output D is BF16.
struct Fp4TestBench {
    // VEC_SIZE = 32 for mxfp4 (one UE8M0 scale per 32 FP4 elements along K)
    static constexpr int VEC_SIZE = 32;

    Fp4TestBench(int m, int n, int k, float alpha = 1.0f, float beta = 0.0f,
                 size_t workspaceSize = 32ULL * 1024 * 1024)
        : m(m), n(n), k(k), alpha(alpha), beta(beta), workspaceSize(workspaceSize) {
        if (k % 2 != 0) throw std::logic_error("K must be even for packed FP4");
        if (k % VEC_SIZE != 0) throw std::logic_error("K must be a multiple of VEC_SIZE");

        // A: (M, K) FP4 packed -> M * K/2 bytes; B: (N, K) FP4 packed -> N * K/2 bytes
        aBytes = (size_t)m * k / 2;
        bBytes = (size_t)n * k / 2;
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

    ~Fp4TestBench() {
        checkHipblasStatus(hipblasLtDestroy(ltHandle));
        checkHipStatus(hipFree(Adev));
        checkHipStatus(hipFree(Bdev));
        checkHipStatus(hipFree(AscaleDev));
        checkHipStatus(hipFree(BscaleDev));
        checkHipStatus(hipFree(Ddev));
        checkHipStatus(hipFree(workspace));
        checkHipStatus(hipStreamDestroy(stream));
    }

    // Two FP4 (E2M1) values packed per byte. 0x2 encodes 1.0 in E2M1
    // (exp=1, mantissa=0), so 0x22 is a pair of ones.
    static constexpr uint8_t FP4_PACKED_FILL = 0x22;
    // E8M0 (UE8M0) encoding of 1.0 is 127 (exponent bias 127, so 2^0 = 1).
    static constexpr uint8_t SCALE_FILL = 127;

    void fillData() {
        std::vector<uint8_t> Ah(aBytes, FP4_PACKED_FILL), Bh(bBytes, FP4_PACKED_FILL);
        std::vector<uint8_t> Asc(aScaleBytes, SCALE_FILL), Bsc(bScaleBytes, SCALE_FILL);

        checkHipStatus(hipMemcpy(Adev, Ah.data(), aBytes, hipMemcpyHostToDevice));
        checkHipStatus(hipMemcpy(Bdev, Bh.data(), bBytes, hipMemcpyHostToDevice));
        checkHipStatus(hipMemcpy(AscaleDev, Asc.data(), aScaleBytes, hipMemcpyHostToDevice));
        checkHipStatus(hipMemcpy(BscaleDev, Bsc.data(), bScaleBytes, hipMemcpyHostToDevice));
    }

    // Decode a 4-bit E2M1 (FP4) code (low nibble) to float.
    static float decodeE2M1(uint8_t code) {
        int sign = (code >> 3) & 0x1;
        int exp  = (code >> 1) & 0x3;
        int man  = code & 0x1;
        float mag;
        if (exp == 0) mag = man * 0.5f;                              // subnormal: 2^0 * man/2
        else          mag = (1.0f + 0.5f * man) * float(1 << (exp - 1));
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

        float aElem  = decodeE2M1(FP4_PACKED_FILL & 0xF);
        float bElem  = decodeE2M1(FP4_PACKED_FILL & 0xF);
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

    void *Adev = nullptr, *Bdev = nullptr;           // packed FP4 (E2M1)
    void *AscaleDev = nullptr, *BscaleDev = nullptr;  // UE8M0 block scales
    __hip_bfloat16 *Ddev = nullptr;                   // BF16 output
    void *workspace = nullptr;
    hipStream_t stream;
    hipblasLtHandle_t ltHandle;
};
