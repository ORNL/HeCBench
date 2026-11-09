#pragma once

#include <cstdint>

// Gaussian evaluation shared by the host reference and every device kernel.
//
// The forward walk stops before a sample that would take the remaining
// transmittance to 1e-4 or below, and the backward kernels replay exactly the
// samples the forward walk kept. That cutoff is an exact comparison, so host
// and device must produce bit-identical sigma, visibility, and transmittance.
// A fused multiply-add or a different exp moves the product by a few ulps and
// can flip the cutoff, which changes which Gaussians a pixel blends.
//
// Every multiply and add below is a single round-to-nearest float operation
// that the compiler cannot fuse. The exp is built from those operations only.

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GSPLAT_FN inline __host__ __device__ __attribute__((always_inline))
#else
#define GSPLAT_FN inline __attribute__((always_inline))
#endif

// AMD device code has no unfusable multiply or add builtin, so it uses the
// VOP2 instruction directly. The host path goes through volatile so that a
// host compiler with FMA enabled cannot contract it either.
GSPLAT_FN float gsplat_mul(float a, float b) {
#if defined(__HIP_DEVICE_COMPILE__) || (defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__))
    float p;
    asm("v_mul_f32_e32 %0, %1, %2" : "=v"(p) : "v"(a), "v"(b));
    return p;
#elif defined(__CUDA_ARCH__)
    return __fmul_rn(a, b);
#elif defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    return __nvvm_mul_rn_f(a, b);
#else
    volatile float x = a;
    volatile float y = b;
    return x * y;
#endif
}

GSPLAT_FN float gsplat_add(float a, float b) {
#if defined(__HIP_DEVICE_COMPILE__) || (defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__))
    float s;
    asm("v_add_f32_e32 %0, %1, %2" : "=v"(s) : "v"(a), "v"(b));
    return s;
#elif defined(__CUDA_ARCH__)
    return __fadd_rn(a, b);
#elif defined(__SYCL_DEVICE_ONLY__) && defined(__NVPTX__)
    return __nvvm_add_rn_f(a, b);
#else
    volatile float x = a;
    volatile float y = b;
    return x + y;
#endif
}

GSPLAT_FN float gsplat_sub(float a, float b) {
    return gsplat_add(a, -b);
}

// Round to nearest even for |y| <= 2^22. Used to pick the exp reduction integer.
GSPLAT_FN float gsplat_rint(float y) {
    const float two23 = 8388608.0f;
    if (y >= 0.f) {
        return gsplat_sub(gsplat_add(y, two23), two23);
    }
    return gsplat_add(gsplat_sub(y, two23), two23);
}

GSPLAT_FN float gsplat_u2f(uint32_t u) {
    union {
        uint32_t u;
        float f;
    } v;
    v.u = u;
    return v.f;
}

GSPLAT_FN uint32_t gsplat_f2u(float f) {
    union {
        uint32_t u;
        float f;
    } v;
    v.f = f;
    return v.u;
}

// exp(x) by Cody-Waite reduction and a degree-9 Taylor polynomial. For every
// float x in [-87.33, 88] the result is within one ulp of the correctly
// rounded exp. Below that range, and for NaN, it returns 0. Above 88 it returns
// 1e30; there x = -sigma means sigma < 0, which every caller skips.
GSPLAT_FN float gsplat_exp(float x) {
    if (!(x > -103.0f)) {
        return 0.f;
    }
    if (x > 88.0f) {
        return 1.0e30f;
    }
    // log2(e), and ln(2) split so n * ln2_hi stays exact for the small n here.
    const float inv_ln2 = 0x1.715476p+0f;
    const float ln2_hi = 0x1.62e000p-1f;
    const float ln2_lo = 0x1.0bfbe8p-15f;
    const float n = gsplat_rint(gsplat_mul(x, inv_ln2));
    float r = gsplat_sub(x, gsplat_mul(n, ln2_hi));
    r = gsplat_sub(r, gsplat_mul(n, ln2_lo));
    // Taylor coefficients 1/9! .. 1/2!, then the final *r + 1.
    float y = 0x1.71de3ap-19f;
    y = gsplat_add(0x1.a01a02p-16f, gsplat_mul(r, y));
    y = gsplat_add(0x1.a01a02p-13f, gsplat_mul(r, y));
    y = gsplat_add(0x1.6c16c2p-10f, gsplat_mul(r, y));
    y = gsplat_add(0x1.111112p-7f, gsplat_mul(r, y));
    y = gsplat_add(0x1.555556p-5f, gsplat_mul(r, y));
    y = gsplat_add(0x1.555556p-3f, gsplat_mul(r, y));
    y = gsplat_add(0.5f, gsplat_mul(r, y));
    y = gsplat_add(1.0f, gsplat_mul(r, y));
    y = gsplat_add(1.0f, gsplat_mul(r, y));
    const int k = static_cast<int>(n);
    const uint32_t bits = gsplat_f2u(y);
    const int exponent = static_cast<int>((bits >> 23) & 0xff) + k;
    if (exponent >= 255) {
        return 1.0e30f;
    }
    if (exponent <= 0) {
        return 0.f;
    }
    return gsplat_u2f((bits & 0x807fffffu) | (static_cast<uint32_t>(exponent) << 23));
}

// sigma, visibility, opacity * visibility, and alpha clamped to 0.999.
struct GsplatEval {
    float sigma;
    float vis;
    float opac_vis;
    float alpha;
};

GSPLAT_FN GsplatEval gsplat_eval(float cxx, float cxy, float cyy, float dx, float dy, float opac) {
    const float dx2 = gsplat_mul(dx, dx);
    const float dy2 = gsplat_mul(dy, dy);
    const float quad = gsplat_mul(0.5f, gsplat_add(gsplat_mul(cxx, dx2), gsplat_mul(cyy, dy2)));
    GsplatEval out;
    out.sigma = gsplat_add(quad, gsplat_mul(cxy, gsplat_mul(dx, dy)));
    out.vis = gsplat_exp(-out.sigma);
    out.opac_vis = gsplat_mul(opac, out.vis);
    out.alpha = out.opac_vis > 0.999f ? 0.999f : out.opac_vis;
    return out;
}

// T * (1 - alpha), rounded once. This is the value compared with 1e-4.
GSPLAT_FN float gsplat_next_T(float transmittance, float alpha) {
    return gsplat_mul(transmittance, gsplat_sub(1.0f, alpha));
}

GSPLAT_FN float gsplat_one_minus(float value) {
    return gsplat_sub(1.0f, value);
}
