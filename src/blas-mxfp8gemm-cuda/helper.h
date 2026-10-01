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
#include <cuda_fp8.h>
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

// Block-scaled MXFP8 GEMM test harness.
//
// Mirrors the cuBLAS mxfp8 path of reference.py:
//   D(M,N) = (A(M,K) * a_scale) @ (B(N,K) * b_scale)^T
// A and B are FP8 (E4M3), one byte per element along K (not packed).
// Scales are E8M0 (UE8M0), one scale per 32-element block along K
// (SCALE_BLOCK_SIZE = 32), matching the OCP microscaling (MX) format.
// Output D is BF16.
struct Mxfp8TestBench {
    // SCALE_BLOCK_SIZE = 32 for mxfp8 (one UE8M0 scale per 32 FP8 elements along K)
    static constexpr int SCALE_BLOCK_SIZE = 32;

    Mxfp8TestBench(int m, int n, int k, float alpha = 1.0f, float beta = 0.0f,
                   size_t workspaceSize = 32ULL * 1024 * 1024)
        : m(m), n(n), k(k), alpha(alpha), beta(beta), workspaceSize(workspaceSize) {
        if (k % SCALE_BLOCK_SIZE != 0) throw std::logic_error("K must be a multiple of SCALE_BLOCK_SIZE");

        // A: (M, K) FP8 -> M * K bytes; B: (N, K) FP8 -> N * K bytes
        aBytes = (size_t)m * k;
        bBytes = (size_t)n * k;
        // One UE8M0 scale (1 byte) per SCALE_BLOCK_SIZE elements along K.
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

    ~Mxfp8TestBench() {
        checkCublasStatus(cublasLtDestroy(ltHandle));
        checkCudaStatus(cudaFree(Adev));
        checkCudaStatus(cudaFree(Bdev));
        checkCudaStatus(cudaFree(AscaleDev));
        checkCudaStatus(cudaFree(BscaleDev));
        checkCudaStatus(cudaFree(Ddev));
        checkCudaStatus(cudaFree(workspace));
        checkCudaStatus(cudaStreamDestroy(stream));
    }

    // FP8 E4M3 encoding of 1.0 is 0x38 (exp bias 7 -> exp field 0b0111, mantissa 0).
    static constexpr uint8_t FP8_FILL = 0x38;
    // E8M0 (UE8M0) encoding of 1.0 is 127 (exponent bias 127, so 2^0 = 1).
    static constexpr uint8_t SCALE_FILL = 127;

    void fillData() {
        std::vector<uint8_t> Ah(aBytes, FP8_FILL), Bh(bBytes, FP8_FILL);
        std::vector<uint8_t> Asc(aScaleBytes, SCALE_FILL), Bsc(bScaleBytes, SCALE_FILL);

        checkCudaStatus(cudaMemcpy(Adev, Ah.data(), aBytes, cudaMemcpyHostToDevice));
        checkCudaStatus(cudaMemcpy(Bdev, Bh.data(), bBytes, cudaMemcpyHostToDevice));
        checkCudaStatus(cudaMemcpy(AscaleDev, Asc.data(), aScaleBytes, cudaMemcpyHostToDevice));
        checkCudaStatus(cudaMemcpy(BscaleDev, Bsc.data(), bScaleBytes, cudaMemcpyHostToDevice));
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
        std::vector<__nv_bfloat16> Dh(dElems);
        checkCudaStatus(cudaMemcpy(Dh.data(), Ddev, dElems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

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
        printf("%s\n", ok ? "PASS" : "FAIL");
        return ok;
    }

    int m, n, k;
    float alpha, beta;
    size_t workspaceSize;
    size_t aBytes, bBytes, aScaleBytes, bScaleBytes, dElems;

    void *Adev = nullptr, *Bdev = nullptr;           // FP8 (E4M3)
    void *AscaleDev = nullptr, *BscaleDev = nullptr;  // UE8M0 block scales
    __nv_bfloat16 *Ddev = nullptr;                    // BF16 output
    void *workspace = nullptr;
    cudaStream_t stream;
    cublasLtHandle_t ltHandle;
};
