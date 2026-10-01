#pragma once

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
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

// Block-scaled MXFP6 (FP6 E2M3) GEMM test harness for CDNA4 (gfx950).
//
// Mirrors the mxfp4/mxfp8 microscaling path:
//   D(M,N) = (A(M,K) * a_scale) @ (B(N,K) * b_scale)^T
// A and B are FP6 (E2M3) packed four elements per three bytes along K.
// Scales are E8M0 (UE8M0), one scale per 32-element block along K (VEC_SIZE = 32),
// as required by the OCP microscaling (MX) format on CDNA4.
// Output D is BF16.
struct Mxfp6TestBench {
    // VEC_SIZE = 32 for mxfp6 (one UE8M0 scale per 32 FP6 elements along K)
    static constexpr int VEC_SIZE = 32;

    Mxfp6TestBench(int m, int n, int k, float alpha = 1.0f, float beta = 0.0f,
                   size_t workspaceSize = 32ULL * 1024 * 1024)
        : m(m), n(n), k(k), alpha(alpha), beta(beta), workspaceSize(workspaceSize) {
        if (k % 4 != 0) throw std::logic_error("K must be a multiple of 4 for packed FP6");
        if (k % VEC_SIZE != 0) throw std::logic_error("K must be a multiple of VEC_SIZE");
        if (k % 128 != 0) throw std::logic_error("K must be a multiple of 128 for MX block scaling");

        // A: (M, K) FP6 packed -> M * (3K/4) bytes; B: (N, K) FP6 packed -> N * (3K/4) bytes
        const size_t kBytes = (size_t)k * 3 / 4;
        aBytes = (size_t)m * kBytes;
        bBytes = (size_t)n * kBytes;
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

    ~Mxfp6TestBench() {
        checkHipblasStatus(hipblasLtDestroy(ltHandle));
        checkHipStatus(hipFree(Adev));
        checkHipStatus(hipFree(Bdev));
        checkHipStatus(hipFree(AscaleDev));
        checkHipStatus(hipFree(BscaleDev));
        checkHipStatus(hipFree(Ddev));
        checkHipStatus(hipFree(workspace));
        checkHipStatus(hipStreamDestroy(stream));
    }

    // FP6 E2M3 encoding of 1.0 is 0x08 (exp=01, mant=000, bias=1).
    static constexpr uint8_t FP6_FILL = 0x08;
    // E8M0 (UE8M0) encoding of 1.0 is 127 (exponent bias 127, so 2^0 = 1).
    static constexpr uint8_t SCALE_FILL = 127;

    static void packUniformFp6(std::vector<uint8_t>& dst, size_t numElements, uint8_t code6) {
        dst.assign(numElements * 3 / 4, 0);
        const uint8_t bits = code6 & 0x3F;
        for (size_t i = 0; i < numElements; ++i) {
            const size_t bit_pos = i * 6;
            const size_t byte_idx = bit_pos / 8;
            const int bit_offset = static_cast<int>(bit_pos % 8);
            dst[byte_idx] |= static_cast<uint8_t>(bits << bit_offset);
            const int overhang = bit_offset + 6 - 8;
            if (overhang > 0 && byte_idx + 1 < dst.size()) {
                dst[byte_idx + 1] |= static_cast<uint8_t>(bits >> (6 - overhang));
            }
        }
    }

    void fillData() {
        std::vector<uint8_t> rowPacked;
        packUniformFp6(rowPacked, (size_t)k, FP6_FILL);

        std::vector<uint8_t> Ah(aBytes), Bh(bBytes);
        for (int row = 0; row < m; ++row) {
            std::memcpy(Ah.data() + (size_t)row * rowPacked.size(), rowPacked.data(), rowPacked.size());
        }
        for (int row = 0; row < n; ++row) {
            std::memcpy(Bh.data() + (size_t)row * rowPacked.size(), rowPacked.data(), rowPacked.size());
        }

        std::vector<uint8_t> Asc(aScaleBytes, SCALE_FILL), Bsc(bScaleBytes, SCALE_FILL);

        checkHipStatus(hipMemcpy(Adev, Ah.data(), aBytes, hipMemcpyHostToDevice));
        checkHipStatus(hipMemcpy(Bdev, Bh.data(), bBytes, hipMemcpyHostToDevice));
        checkHipStatus(hipMemcpy(AscaleDev, Asc.data(), aScaleBytes, hipMemcpyHostToDevice));
        checkHipStatus(hipMemcpy(BscaleDev, Bsc.data(), bScaleBytes, hipMemcpyHostToDevice));
    }

    // Decode a 6-bit E2M3 (FP6) code to float (exponent bias 1).
    static float decodeE2M3(uint8_t code) {
        code &= 0x3F;
        int sign = (code >> 5) & 0x1;
        int exp  = (code >> 3) & 0x3;
        int man  = code & 0x7;
        float mag;
        if (exp == 0) mag = (man / 8.0f) * 0.5f;
        else          mag = (1.0f + man / 8.0f) * float(1 << (exp - 1));
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

        float aElem  = decodeE2M3(FP6_FILL);
        float bElem  = decodeE2M3(FP6_FILL);
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

    void *Adev = nullptr, *Bdev = nullptr;           // packed FP6 (E2M3)
    void *AscaleDev = nullptr, *BscaleDev = nullptr;  // UE8M0 block scales
    __hip_bfloat16 *Ddev = nullptr;                   // BF16 output
    void *workspace = nullptr;
    hipStream_t stream;
    hipblasLtHandle_t ltHandle;
};
