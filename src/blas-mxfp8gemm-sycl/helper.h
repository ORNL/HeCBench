#pragma once

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <vector>

#include <sycl/sycl.hpp>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

// Block-scaled MXFP8 GEMM test harness (SYCL + oneDNN).
//
// Mirrors the cuBLASLt/hipBLASLt mxfp8 path of blas-mxfp8gemm-cuda:
//   D(M,N) = (A(M,K) * a_scale) @ (B(N,K) * b_scale)^T
// A and B are FP8 (E4M3), one byte per element along K (not packed).
// Scales are E8M0 (UE8M0), one scale per 32-element block along K
// (SCALE_BLOCK_SIZE = 32), matching the OCP microscaling (MX) format.
// Output D is BF16.
//
// The A / B elements and their block scales are supplied by the caller
// (already quantized), which matches the oneDNN mxfp8 matmul example where the
// SRC and WEIGHTS scales are user-provided E8M0 tensors.
struct Mxfp8TestBench {
    // SCALE_BLOCK_SIZE = 32 for mxfp8 (one UE8M0 scale per 32 FP8 elements along K)
    static constexpr int SCALE_BLOCK_SIZE = 32;

    // FP8 E4M3 encoding of 1.0 is 0x38 (exp bias 7 -> exp field 0b0111, mantissa 0).
    static constexpr uint8_t FP8_FILL = 0x38;
    // E8M0 (UE8M0) encoding of 1.0 is 127 (exponent bias 127, so 2^0 = 1).
    static constexpr uint8_t SCALE_FILL = 127;

    Mxfp8TestBench(sycl::queue &queue, int m, int n, int k,
                   float alpha = 1.0f, float beta = 0.0f)
        : q(queue),
          engine(dnnl::sycl_interop::make_engine(q.get_device(), q.get_context())),
          stream(dnnl::sycl_interop::make_stream(engine, q)),
          m(m), n(n), k(k), alpha(alpha), beta(beta) {
        if (k % SCALE_BLOCK_SIZE != 0) throw std::logic_error("K must be a multiple of SCALE_BLOCK_SIZE");

        // A: (M, K) FP8 -> M * K bytes; B: (N, K) FP8 -> N * K bytes
        aBytes = (size_t)m * k;
        bBytes = (size_t)n * k;
        // One UE8M0 scale (1 byte) per SCALE_BLOCK_SIZE elements along K.
        aScaleBytes = (size_t)m * (k / SCALE_BLOCK_SIZE);
        bScaleBytes = (size_t)n * (k / SCALE_BLOCK_SIZE);
        dElems = (size_t)m * n;

        Adev = sycl::malloc_device<uint8_t>(aBytes, q);
        Bdev = sycl::malloc_device<uint8_t>(bBytes, q);
        AscaleDev = sycl::malloc_device<uint8_t>(aScaleBytes, q);
        BscaleDev = sycl::malloc_device<uint8_t>(bScaleBytes, q);
        Ddev = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(dElems, q);

        fillData();
    }

    ~Mxfp8TestBench() {
        sycl::free(Adev, q);
        sycl::free(Bdev, q);
        sycl::free(AscaleDev, q);
        sycl::free(BscaleDev, q);
        sycl::free(Ddev, q);
    }

    void fillData() {
        std::vector<uint8_t> Ah(aBytes, FP8_FILL), Bh(bBytes, FP8_FILL);
        std::vector<uint8_t> Asc(aScaleBytes, SCALE_FILL), Bsc(bScaleBytes, SCALE_FILL);

        q.memcpy(Adev, Ah.data(), aBytes);
        q.memcpy(Bdev, Bh.data(), bBytes);
        q.memcpy(AscaleDev, Asc.data(), aScaleBytes);
        q.memcpy(BscaleDev, Bsc.data(), bScaleBytes);
        q.wait();
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
        std::vector<sycl::ext::oneapi::bfloat16> Dh(dElems);
        q.memcpy(Dh.data(), Ddev, dElems * sizeof(sycl::ext::oneapi::bfloat16)).wait();

        float aElem  = decodeE4M3(FP8_FILL);
        float bElem  = decodeE4M3(FP8_FILL);
        float aScale = decodeUE8M0(SCALE_FILL);
        float bScale = decodeUE8M0(SCALE_FILL);
        double expected = (double)alpha * (double)(aElem * aScale) *
                          (double)(bElem * bScale) * (double)k;

        double maxErr = 0.0;
        for (size_t i = 0; i < dElems; i++) {
            double got = (double)(float)Dh[i];
            maxErr = std::max(maxErr, std::fabs(got - expected));
        }
        bool ok = maxErr <= relTol * (std::fabs(expected) + 1.0);
        printf("%s\n", ok ? "PASS" : "FAIL");
        return ok;
    }

    sycl::queue &q;
    dnnl::engine engine;
    dnnl::stream stream;

    int m, n, k;
    float alpha, beta;
    size_t aBytes, bBytes, aScaleBytes, bScaleBytes, dElems;

    uint8_t *Adev = nullptr, *Bdev = nullptr;          // FP8 (E4M3)
    uint8_t *AscaleDev = nullptr, *BscaleDev = nullptr; // UE8M0 block scales
    sycl::ext::oneapi::bfloat16 *Ddev = nullptr;        // BF16 output
};
