#pragma once

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <vector>

#include <cublasLt.h>
//#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

inline void checkCudaStatus(cudaError_t status) {
    if (status != cudaSuccess) {
        printf("CUDA API failed with status %d: %s\n", status, cudaGetErrorString(status));
        throw std::logic_error("cuda API failed");
    }
}

inline void checkCublasStatus(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("cuBLAS API failed with status %d\n", status);
        throw std::logic_error("cuBLAS API failed");
    }
}

// Block-scaled FP4 (nvfp4) GEMM test harness.
//
// Mirrors the cuBLAS nvfp4 path of reference.py:
//   D(M,N) = (A(M,K) * a_scale) @ (B(N,K) * b_scale)^T
// A and B are FP4 (E2M1) packed two elements per byte along K.
// Scales are UE4M3, one scale per 16-element block along K (SCALE_BLOCK_SIZE = 16).
// Output D is BF16.
struct Fp4TestBench {
    // SCALE_BLOCK_SIZE = 16 for nvfp4 (one UE4M3 scale per 16 FP4 elements along K)
    static constexpr int SCALE_BLOCK_SIZE = 16;

    Fp4TestBench(int m, int n, int k, float alpha = 1.0f, float beta = 0.0f,
                 size_t workspaceSize = 32ULL * 1024 * 1024)
        : m(m), n(n), k(k), alpha(alpha), beta(beta), workspaceSize(workspaceSize) {
        if (k % 2 != 0) throw std::logic_error("K must be even for packed FP4");
        if (k % SCALE_BLOCK_SIZE != 0) throw std::logic_error("K must be a multiple of SCALE_BLOCK_SIZE");

        // A: (M, K) FP4 packed -> M * K/2 bytes; B: (N, K) FP4 packed -> N * K/2 bytes
        aBytes = (size_t)m * k / 2;
        bBytes = (size_t)n * k / 2;
        // One UE4M3 scale (1 byte) per SCALE_BLOCK_SIZE elements along K.
        aScaleBytes = (size_t)m * (k / SCALE_BLOCK_SIZE);
        bScaleBytes = (size_t)n * (k / SCALE_BLOCK_SIZE);
        dElems = (size_t)m * n;

        checkCublasStatus(cublasLtCreate(&ltHandle));
        checkCudaStatus(cudaMalloc(&Adev, aBytes));
        checkCudaStatus(cudaMalloc(&Bdev, bBytes));
        checkCudaStatus(cudaMalloc(&AscaleDev, aScaleBytes));
        checkCudaStatus(cudaMalloc(&BscaleDev, bScaleBytes));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Ddev), dElems * sizeof(__nv_bfloat16)));
        checkCudaStatus(cudaMalloc(&workspace, workspaceSize));
        checkCudaStatus(cudaStreamCreate(&stream));

        fillData();
    }

    ~Fp4TestBench() {
        checkCublasStatus(cublasLtDestroy(ltHandle));
        checkCudaStatus(cudaFree(Adev));
        checkCudaStatus(cudaFree(Bdev));
        checkCudaStatus(cudaFree(AscaleDev));
        checkCudaStatus(cudaFree(BscaleDev));
        checkCudaStatus(cudaFree(Ddev));
        checkCudaStatus(cudaFree(workspace));
        checkCudaStatus(cudaStreamDestroy(stream));
    }

    // Two FP4 (E2M1) values packed per byte. 0x2 encodes 1.0 in E2M1
    // (exp=1, mantissa=0), so 0x22 is a pair of ones.
    static constexpr uint8_t FP4_PACKED_FILL = 0x22;
    // UE4M3 encoding of 1.0 is 0x38 (exp bias 7 -> exp field 0b0111, mantissa 0).
    static constexpr uint8_t SCALE_FILL = 0x38;

    void fillData() {
        std::vector<uint8_t> Ah(aBytes, FP4_PACKED_FILL), Bh(bBytes, FP4_PACKED_FILL);
        std::vector<uint8_t> Asc(aScaleBytes, SCALE_FILL), Bsc(bScaleBytes, SCALE_FILL);

        checkCudaStatus(cudaMemcpy(Adev, Ah.data(), aBytes, cudaMemcpyHostToDevice));
        checkCudaStatus(cudaMemcpy(Bdev, Bh.data(), bBytes, cudaMemcpyHostToDevice));
        checkCudaStatus(cudaMemcpy(AscaleDev, Asc.data(), aScaleBytes, cudaMemcpyHostToDevice));
        checkCudaStatus(cudaMemcpy(BscaleDev, Bsc.data(), bScaleBytes, cudaMemcpyHostToDevice));
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

    // Decode an 8-bit UE4M3 scale code to float (E4M3 layout, exponent bias 7).
    static float decodeUE4M3(uint8_t code) {
        int exp = (code >> 3) & 0xF;
        int man = code & 0x7;
        if (exp == 0) return (man / 8.0f) * ldexpf(1.0f, 1 - 7);     // subnormal
        return (1.0f + man / 8.0f) * ldexpf(1.0f, exp - 7);
    }

    // CPU reference check. The inputs are uniform, so every output element of
    //   D = alpha * (A*a_scale) @ (B*b_scale)^T + beta * C
    // equals alpha * (a_elem*a_scale) * (b_elem*b_scale) * K (with beta = 0).
    bool verify(double relTol = 1e-2) {
        std::vector<__nv_bfloat16> Dh(dElems);
        checkCudaStatus(cudaMemcpy(Dh.data(), Ddev, dElems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

        float aElem  = decodeE2M1(FP4_PACKED_FILL & 0xF);
        float bElem  = decodeE2M1(FP4_PACKED_FILL & 0xF);
        float aScale = decodeUE4M3(SCALE_FILL);
        float bScale = decodeUE4M3(SCALE_FILL);
        double expected = (double)alpha * (double)(aElem * aScale) *
                          (double)(bElem * bScale) * (double)k;

        double maxErr = 0.0;
        for (size_t i = 0; i < dElems; i++) {
            double got = (double)__bfloat162float(Dh[i]);
            maxErr = std::max(maxErr, std::fabs(got - expected));
        }
        bool ok = maxErr <= relTol * (std::fabs(expected) + 1.0);
        //printf("Verification: expected %.1f, max abs error %.4g -> %s\n",
        //       expected, maxErr, ok ? "PASS" : "FAIL");
        printf("%s\n", ok ? "PASS" : "FAIL");
        return ok;
    }

    int m, n, k;
    float alpha, beta;
    size_t workspaceSize;
    size_t aBytes, bBytes, aScaleBytes, bScaleBytes, dElems;

    void *Adev = nullptr, *Bdev = nullptr;          // packed FP4 (E2M1)
    void *AscaleDev = nullptr, *BscaleDev = nullptr; // UE4M3 block scales
    __nv_bfloat16 *Ddev = nullptr;                   // BF16 output
    void *workspace = nullptr;
    cudaStream_t stream;
    cublasLtHandle_t ltHandle;
};
