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

// Per-tensor scaled FP8 GEMM test harness (SYCL + oneDNN).
//
// D(M,N) = (A(M,K) * a_scale) @ (B(N,K) * b_scale)^T
// A and B are FP8 (E4M3), one byte per element along K (not packed).
// Unlike the mxfp8 variant there is a single FP32 scale per tensor
// (per-tensor scaling), not one E8M0 scale per 32-element block.
// Output D is BF16.
//
// The A / B elements and their per-tensor scales are supplied by the caller
// (already quantized), which matches the oneDNN fp8 matmul path where the
// SRC and WEIGHTS scales are user-provided FP32 tensors.
struct Fp8TestBench {
    // FP8 E4M3 encoding of 1.0 is 0x38 (exp bias 7 -> exp field 0b0111, mantissa 0).
    static constexpr uint8_t FP8_FILL = 0x38;

    Fp8TestBench(sycl::queue &queue, int m, int n, int k,
                 float alpha = 1.0f, float beta = 0.0f,
                 float aScale = 1.0f, float bScale = 1.0f)
        : q(queue),
          engine(dnnl::sycl_interop::make_engine(q.get_device(), q.get_context())),
          stream(dnnl::sycl_interop::make_stream(engine, q)),
          m(m), n(n), k(k), alpha(alpha), beta(beta),
          aScale(aScale), bScale(bScale) {

        // A: (M, K) FP8 -> M * K bytes; B: (N, K) FP8 -> N * K bytes
        aBytes = (size_t)m * k;
        bBytes = (size_t)n * k;
        dElems = (size_t)m * n;

        Adev = sycl::malloc_device<uint8_t>(aBytes, q);
        Bdev = sycl::malloc_device<uint8_t>(bBytes, q);
        AscaleDev = sycl::malloc_device<float>(1, q);
        BscaleDev = sycl::malloc_device<float>(1, q);
        Ddev = sycl::malloc_device<sycl::ext::oneapi::bfloat16>(dElems, q);

        fillData();
    }

    ~Fp8TestBench() {
        sycl::free(Adev, q);
        sycl::free(Bdev, q);
        sycl::free(AscaleDev, q);
        sycl::free(BscaleDev, q);
        sycl::free(Ddev, q);
    }

    void fillData() {
        std::vector<uint8_t> Ah(aBytes, FP8_FILL), Bh(bBytes, FP8_FILL);

        q.memcpy(Adev, Ah.data(), aBytes);
        q.memcpy(Bdev, Bh.data(), bBytes);
        q.memcpy(AscaleDev, &aScale, sizeof(float));
        q.memcpy(BscaleDev, &bScale, sizeof(float));
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

    // CPU reference check. The inputs are uniform, so every output element of
    //   D = alpha * (A*a_scale) @ (B*b_scale)^T + beta * C
    // equals alpha * (a_elem*a_scale) * (b_elem*b_scale) * K (with beta = 0).
    bool verify(double relTol = 1e-2) {
        std::vector<sycl::ext::oneapi::bfloat16> Dh(dElems);
        q.memcpy(Dh.data(), Ddev, dElems * sizeof(sycl::ext::oneapi::bfloat16)).wait();

        float aElem  = decodeE4M3(FP8_FILL);
        float bElem  = decodeE4M3(FP8_FILL);
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
    float aScale, bScale;
    size_t aBytes, bBytes, dElems;

    uint8_t *Adev = nullptr, *Bdev = nullptr;   // FP8 (E4M3)
    float *AscaleDev = nullptr, *BscaleDev = nullptr; // FP32 per-tensor scales
    sycl::ext::oneapi::bfloat16 *Ddev = nullptr; // BF16 output
};
