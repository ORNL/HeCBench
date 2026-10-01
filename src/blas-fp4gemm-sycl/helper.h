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

// Block-scaled MXFP4 GEMM test harness (SYCL + oneDNN).
//
//   D(M,N) = (A(M,K) * a_scale) @ (B(N,K) * b_scale)^T
// A and B are FP4 (E2M1) packed two elements per byte along K.
// Scales are E8M0 (UE8M0), one scale per 32-element block along K
// (SCALE_BLOCK_SIZE = 32), matching the OCP MX format (mxfp4).
// Output D is BF16.
//
// The A / B elements and their block scales are supplied by the caller
// (already quantized), which matches the oneDNN mxfp matmul example where the
// SRC and WEIGHTS scales are user-provided E8M0 tensors.
struct Mxfp4TestBench {
    // SCALE_BLOCK_SIZE = 32 for mxfp4 (one E8M0 scale per 32 FP4 elements along K)
    static constexpr int SCALE_BLOCK_SIZE = 32;

    // Two FP4 (E2M1) values packed per byte. 0x2 encodes 1.0 in E2M1
    // (exp=1, mantissa=0), so 0x22 is a pair of ones.
    static constexpr uint8_t FP4_PACKED_FILL = 0x22;
    // E8M0 (UE8M0) encoding of 1.0 is 127 (exponent bias 127, so 2^0 = 1).
    static constexpr uint8_t SCALE_FILL = 127;

    Mxfp4TestBench(sycl::queue &queue, int m, int n, int k,
                   float alpha = 1.0f, float beta = 0.0f)
        : q(queue),
          engine(dnnl::sycl_interop::make_engine(q.get_device(), q.get_context())),
          stream(dnnl::sycl_interop::make_stream(engine, q)),
          m(m), n(n), k(k), alpha(alpha), beta(beta) {
        if (k % 2 != 0) throw std::logic_error("K must be even for packed FP4");
        if (k % SCALE_BLOCK_SIZE != 0) throw std::logic_error("K must be a multiple of SCALE_BLOCK_SIZE");

        // A: (M, K) FP4 packed -> M * K/2 bytes; B: (N, K) FP4 packed -> N * K/2 bytes
        aBytes = (size_t)m * k / 2;
        bBytes = (size_t)n * k / 2;
        // One E8M0 scale (1 byte) per SCALE_BLOCK_SIZE elements along K.
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

    ~Mxfp4TestBench() {
        sycl::free(Adev, q);
        sycl::free(Bdev, q);
        sycl::free(AscaleDev, q);
        sycl::free(BscaleDev, q);
        sycl::free(Ddev, q);
    }

    void fillData() {
        std::vector<uint8_t> Ah(aBytes, FP4_PACKED_FILL), Bh(bBytes, FP4_PACKED_FILL);
        std::vector<uint8_t> Asc(aScaleBytes, SCALE_FILL), Bsc(bScaleBytes, SCALE_FILL);

        q.memcpy(Adev, Ah.data(), aBytes);
        q.memcpy(Bdev, Bh.data(), bBytes);
        q.memcpy(AscaleDev, Asc.data(), aScaleBytes);
        q.memcpy(BscaleDev, Bsc.data(), bScaleBytes);
        q.wait();
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
        std::vector<sycl::ext::oneapi::bfloat16> Dh(dElems);
        q.memcpy(Dh.data(), Ddev, dElems * sizeof(sycl::ext::oneapi::bfloat16)).wait();

        float aElem  = decodeE2M1(FP4_PACKED_FILL & 0xF);
        float bElem  = decodeE2M1(FP4_PACKED_FILL & 0xF);
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

    uint8_t *Adev = nullptr, *Bdev = nullptr;          // packed FP4 (E2M1)
    uint8_t *AscaleDev = nullptr, *BscaleDev = nullptr; // E8M0 block scales
    sycl::ext::oneapi::bfloat16 *Ddev = nullptr;        // BF16 output
};
