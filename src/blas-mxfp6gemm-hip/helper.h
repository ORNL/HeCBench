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
// stored row-major. hipBLASLt reads VEC32_UE8M0 scales in that order.
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
        // One UE8M0 scale (1 byte) per VEC_SIZE elements along K, row-major.
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

        fillRandom();
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

    // Random FP6 codes (all 64 E2M3 codes are finite) and UE8M0 scales in [2^-3, 2^3].
    // Host copies of the unpacked codes and scales are kept for the reference check.
    // Scales are written row-major: index (row, block) is at row * numBlocks + block.
    void fillRandom() {
        std::mt19937 gen(19937);
        std::uniform_int_distribution<int> code6(0, 63), scale(124, 130);
        const int numBlocks = k / VEC_SIZE;

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

        checkHipStatus(hipMemcpy(Adev, Ah.data(), aBytes, hipMemcpyHostToDevice));
        checkHipStatus(hipMemcpy(Bdev, Bh.data(), bBytes, hipMemcpyHostToDevice));
        checkHipStatus(hipMemcpy(AscaleDev, Ascales.data(), aScaleBytes, hipMemcpyHostToDevice));
        checkHipStatus(hipMemcpy(BscaleDev, Bscales.data(), bScaleBytes, hipMemcpyHostToDevice));
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
        const int numBlocks = k / VEC_SIZE;
        double sum = 0.0;
        absSum = 0.0;
        for (int kk = 0; kk < k; ++kk) {
            const double a = (double)decodeE2M3(Acodes[(size_t)i * k + kk]) *
                             decodeUE8M0(Ascales[(size_t)i * numBlocks + kk / VEC_SIZE]);
            const double b = (double)decodeE2M3(Bcodes[(size_t)j * k + kk]) *
                             decodeUE8M0(Bscales[(size_t)j * numBlocks + kk / VEC_SIZE]);
            sum += a * b;
            absSum += std::fabs(a * b);
        }
        absSum *= std::fabs(alpha);
        return alpha * sum;
    }

    // Checks the border, a stride-64 interior grid, and 2048 random outputs
    // against refDot. D is column-major (M,N) with ldd = M.
    //
    // relTol is two BF16 ulps (2^-6). One ulp is the BF16 rounding of the FP32
    // accumulator; the second covers that result sitting off the exact dot
    // product. accTol is a few sqrt(K) FP32 ulps of sum|term|. A fixed 0.1% of
    // sum|term| hid a several-percent scale error when |ref| << sum|term|.
    bool verify() {
        std::vector<__hip_bfloat16> Dh(dElems);
        checkHipStatus(hipMemcpy(Dh.data(), Ddev, dElems * sizeof(__hip_bfloat16), hipMemcpyDeviceToHost));

        const double relTol = 0x1p-6;
        const double accTol = 4.0 * std::sqrt(static_cast<double>(k)) * 0x1p-23;
        constexpr int stride = 64;
        constexpr int randomSamples = 2048;

        int bad = 0;
        int checked = 0;
        double worst = 0.0;
        double worstRel = 0.0;
        double worstAcc = 0.0;
        auto check = [&](int i, int j) {
            double absSum;
            const double ref = refDot(i, j, absSum);
            const double got = (double)__bfloat162float(Dh[(size_t)j * m + i]);
            const double err = std::fabs(got - ref);
            const double tol = relTol * std::fabs(ref) + accTol * absSum + 1e-6;
            const double ratio = err / tol;
            worst = std::max(worst, ratio);
            worstRel = std::max(worstRel, err / (std::fabs(ref) + 1e-30));
            worstAcc = std::max(worstAcc, err / (absSum + 1e-30));
            checked++;
            if (ratio > 1.0) {
                if (bad < 5) printf("Mismatch at (%d, %d): got %g, expected %g\n", i, j, got, ref);
                bad++;
            }
        };

        for (int i = 0; i < m; ++i) {
            check(i, 0);
            if (n > 1) check(i, n - 1);
        }
        for (int j = 1; j + 1 < n; ++j) {
            check(0, j);
            if (m > 1) check(m - 1, j);
        }
        for (int i = stride; i < m - 1; i += stride)
            for (int j = stride; j < n - 1; j += stride)
                check(i, j);

        std::mt19937 gen(4321);
        std::uniform_int_distribution<int> rowDist(0, m - 1), colDist(0, n - 1);
        for (int s = 0; s < randomSamples; ++s) check(rowDist(gen), colDist(gen));

        const bool ok = (bad == 0);
        printf("Verification: %d of %d checks out of tolerance (border + stride-%d grid + %d random), "
               "worst rel %.3g, worst |err|/sum|term| %.3g, worst error/tolerance %.3f -> %s\n",
               bad, checked, stride, randomSamples, worstRel, worstAcc, worst, ok ? "PASS" : "FAIL");
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
    __hip_bfloat16 *Ddev = nullptr;                   // BF16 output
    void *workspace = nullptr;
    hipStream_t stream;
    hipblasLtHandle_t ltHandle;
};
