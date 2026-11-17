#pragma once

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <random>
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

// Block-scaled MXFP6 (FP6 E2M3) GEMM test harness.
//
// Mirrors the cuBLAS mxfp6 path of reference.py:
//   D(M,N) = (A(M,K) * a_scale) @ (B(N,K) * b_scale)^T
// A and B are FP6 (E2M3) packed four elements per three bytes along K.
// Scales are E8M0 (UE8M0), one scale per 32-element block along K
// (SCALE_BLOCK_SIZE = 32), matching the OCP microscaling (MX) format.
// Output D is BF16.
struct Mxfp6TestBench {
    // SCALE_BLOCK_SIZE = 32 for mxfp6 (one UE8M0 scale per 32 FP6 elements along K)
    static constexpr int SCALE_BLOCK_SIZE = 32;

    Mxfp6TestBench(int m, int n, int k, float alpha = 1.0f, float beta = 0.0f,
                   size_t workspaceSize = 32ULL * 1024 * 1024)
        : m(m), n(n), k(k), alpha(alpha), beta(beta), workspaceSize(workspaceSize) {
        if (k % 4 != 0) throw std::logic_error("K must be a multiple of 4 for packed FP6");
        if (k % SCALE_BLOCK_SIZE != 0) throw std::logic_error("K must be a multiple of SCALE_BLOCK_SIZE");
        if (k % 128 != 0) throw std::logic_error("K must be a multiple of 128 for MX block scaling");

        // A: (M, K) FP6 packed -> M * (3K/4) bytes; B: (N, K) FP6 packed -> N * (3K/4) bytes
        const size_t kBytes = (size_t)k * 3 / 4;
        aBytes = (size_t)m * kBytes;
        bBytes = (size_t)n * kBytes;
        // One UE8M0 scale (1 byte) per SCALE_BLOCK_SIZE elements along K,
        // stored in 128-row x 4-block tiles (see scaleOffset).
        const size_t scaleCols = (size_t)(k / SCALE_BLOCK_SIZE + 3) / 4 * 4;
        aScaleBytes = (size_t)(m + 127) / 128 * 128 * scaleCols;
        bScaleBytes = (size_t)(n + 127) / 128 * 128 * scaleCols;
        dElems = (size_t)m * n;

        checkCublasStatus(cublasLtCreate(&ltHandle));
        checkCudaStatus(cudaMalloc(&Adev, aBytes));
        checkCudaStatus(cudaMalloc(&Bdev, bBytes));
        checkCudaStatus(cudaMalloc(&AscaleDev, aScaleBytes));
        checkCudaStatus(cudaMalloc(&BscaleDev, bScaleBytes));
        checkCudaStatus(cudaMalloc(reinterpret_cast<void**>(&Ddev), dElems * sizeof(__nv_bfloat16)));
        checkCudaStatus(cudaMalloc(&workspace, workspaceSize));
        checkCudaStatus(cudaStreamCreate(&stream));

        fillRandom();
    }

    ~Mxfp6TestBench() {
        checkCublasStatus(cublasLtDestroy(ltHandle));
        checkCudaStatus(cudaFree(Adev));
        checkCudaStatus(cudaFree(Bdev));
        checkCudaStatus(cudaFree(AscaleDev));
        checkCudaStatus(cudaFree(BscaleDev));
        checkCudaStatus(cudaFree(Ddev));
        checkCudaStatus(cudaFree(workspace));
        checkCudaStatus(cudaStreamDestroy(stream));
    }

    // Packs 6-bit codes four elements per three bytes, least-significant bits first.
    static void packFp6(std::vector<uint8_t>& dst, const std::vector<uint8_t>& codes) {
        dst.assign(codes.size() * 3 / 4, 0);
        for (size_t i = 0; i < codes.size(); ++i) {
            const uint8_t bits = codes[i] & 0x3F;
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

    // Byte offset of the scale for (row, block) in the cuBLASLt VEC32_UE8M0 layout:
    // 512-byte tiles of 128 rows x 4 blocks, tiles row-major over the matrix.
    static size_t scaleOffset(int row, int blk, int numBlocks) {
        const int tileCols = (numBlocks + 3) / 4;
        return ((size_t)(row / 128) * tileCols + blk / 4) * 512 +
               (row % 32) * 16 + ((row % 128) / 32) * 4 + (blk % 4);
    }

    // Random FP6 codes (all 64 E2M3 codes are finite) and UE8M0 scales in [2^-3, 2^3].
    // Host copies of the unpacked codes and scales are kept for the reference check.
    void fillRandom() {
        std::mt19937 gen(19937);
        std::uniform_int_distribution<int> code6(0, 63), scale(124, 130);
        const int numBlocks = k / SCALE_BLOCK_SIZE;

        Acodes.resize((size_t)m * k);
        Bcodes.resize((size_t)n * k);
        for (auto &c : Acodes) c = (uint8_t)code6(gen);
        for (auto &c : Bcodes) c = (uint8_t)code6(gen);
        Ascales.resize((size_t)m * numBlocks);
        Bscales.resize((size_t)n * numBlocks);
        for (auto &s : Ascales) s = (uint8_t)scale(gen);
        for (auto &s : Bscales) s = (uint8_t)scale(gen);

        std::vector<uint8_t> Ah, Bh;
        packFp6(Ah, Acodes);
        packFp6(Bh, Bcodes);

        std::vector<uint8_t> Asc(aScaleBytes, 127), Bsc(bScaleBytes, 127);
        for (int row = 0; row < m; ++row)
            for (int b = 0; b < numBlocks; ++b)
                Asc[scaleOffset(row, b, numBlocks)] = Ascales[(size_t)row * numBlocks + b];
        for (int row = 0; row < n; ++row)
            for (int b = 0; b < numBlocks; ++b)
                Bsc[scaleOffset(row, b, numBlocks)] = Bscales[(size_t)row * numBlocks + b];

        checkCudaStatus(cudaMemcpy(Adev, Ah.data(), aBytes, cudaMemcpyHostToDevice));
        checkCudaStatus(cudaMemcpy(Bdev, Bh.data(), bBytes, cudaMemcpyHostToDevice));
        checkCudaStatus(cudaMemcpy(AscaleDev, Asc.data(), aScaleBytes, cudaMemcpyHostToDevice));
        checkCudaStatus(cudaMemcpy(BscaleDev, Bsc.data(), bScaleBytes, cudaMemcpyHostToDevice));
    }

    // Decode a 6-bit E2M3 (FP6) code to float (exponent bias 1).
    static float decodeE2M3(uint8_t code) {
        code &= 0x3F;
        int sign = (code >> 5) & 0x1;
        int exp  = (code >> 3) & 0x3;
        int man  = code & 0x7;
        float mag;
        if (exp == 0) mag = man / 8.0f;
        else          mag = (1.0f + man / 8.0f) * float(1 << (exp - 1));
        return sign ? -mag : mag;
    }

    // Decode an 8-bit E8M0 (UE8M0) scale code to float: value = 2^(code - 127).
    static float decodeUE8M0(uint8_t code) {
        return ldexpf(1.0f, (int)code - 127);
    }

    // Double-precision reference for D(i,j) = alpha * sum_k (A(i,k)*a_scale) * (B(j,k)*b_scale)
    // (beta = 0). absSum receives sum_k |term|, which bounds the FP32 accumulation error.
    double refDot(int i, int j, double &absSum) const {
        const int numBlocks = k / SCALE_BLOCK_SIZE;
        double sum = 0.0;
        absSum = 0.0;
        for (int kk = 0; kk < k; ++kk) {
            const double a = (double)decodeE2M3(Acodes[(size_t)i * k + kk]) *
                             decodeUE8M0(Ascales[(size_t)i * numBlocks + kk / SCALE_BLOCK_SIZE]);
            const double b = (double)decodeE2M3(Bcodes[(size_t)j * k + kk]) *
                             decodeUE8M0(Bscales[(size_t)j * numBlocks + kk / SCALE_BLOCK_SIZE]);
            sum += a * b;
            absSum += std::fabs(a * b);
        }
        absSum *= std::fabs(alpha);
        return alpha * sum;
    }

    // Checks sampled outputs (including the four corners) against refDot. D is
    // column-major (M,N) with ldd = M. The tolerance covers BF16 output rounding
    // (relTol * |ref|) and FP32 accumulation (accTol * sum|term|).
    bool verify(int samples = 2048, double relTol = 1e-2, double accTol = 1e-3) {
        std::vector<__nv_bfloat16> Dh(dElems);
        checkCudaStatus(cudaMemcpy(Dh.data(), Ddev, dElems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));

        std::mt19937 gen(4321);
        std::uniform_int_distribution<int> rowDist(0, m - 1), colDist(0, n - 1);
        int bad = 0;
        double worst = 0.0;
        for (int s = 0; s < samples; ++s) {
            int i, j;
            if (s < 4) { i = (s & 1) ? m - 1 : 0; j = (s & 2) ? n - 1 : 0; }
            else       { i = rowDist(gen); j = colDist(gen); }
            double absSum;
            const double ref = refDot(i, j, absSum);
            const double got = (double)__bfloat162float(Dh[(size_t)j * m + i]);
            const double tol = relTol * std::fabs(ref) + accTol * absSum + 1e-6;
            const double ratio = std::fabs(got - ref) / tol;
            worst = std::max(worst, ratio);
            if (ratio > 1.0) {
                if (bad < 5) printf("Mismatch at (%d, %d): got %g, expected %g\n", i, j, got, ref);
                bad++;
            }
        }
        const bool ok = (bad == 0);
        printf("Verification: %d of %d sampled outputs out of tolerance, worst error/tolerance %.3f -> %s\n",
               bad, samples, worst, ok ? "PASS" : "FAIL");
        return ok;
    }

    int m, n, k;
    float alpha, beta;
    size_t workspaceSize;
    size_t aBytes, bBytes, aScaleBytes, bScaleBytes, dElems;

    std::vector<uint8_t> Acodes, Bcodes;    // unpacked FP6 codes, (M,K) and (N,K) row-major
    std::vector<uint8_t> Ascales, Bscales;  // UE8M0 codes, (M,K/32) and (N,K/32) row-major

    void *Adev = nullptr, *Bdev = nullptr;           // packed FP6 (E2M3)
    void *AscaleDev = nullptr, *BscaleDev = nullptr;  // UE8M0 block scales
    __nv_bfloat16 *Ddev = nullptr;                    // BF16 output
    void *workspace = nullptr;
    cudaStream_t stream;
    cublasLtHandle_t ltHandle;
};
