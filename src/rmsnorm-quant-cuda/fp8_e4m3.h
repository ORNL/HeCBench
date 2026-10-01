// Portable E4M3 encode/decode for OCP (bias 7, max 448) and FNUZ (bias 8, max 240).
// Matches the host reference used by addBiasQKV.

#ifndef RMSNORM_QUANT_FP8_E4M3_H
#define RMSNORM_QUANT_FP8_E4M3_H

#include <cstdint>
#include <cmath>
#include <cstring>

#if defined(SYCL_LANGUAGE_VERSION)
#include <sycl/sycl.hpp>
#endif

#if defined(__CUDACC__) || defined(__HIPCC__)
#define RQ_HD __host__ __device__
#else
#define RQ_HD
#endif

namespace rq {

RQ_HD inline float bits_to_f32(uint32_t b) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return __uint_as_float(b);
#elif defined(SYCL_LANGUAGE_VERSION)
  return sycl::bit_cast<float>(b);
#else
  float f;
  std::memcpy(&f, &b, sizeof(f));
  return f;
#endif
}

RQ_HD inline uint32_t f32_to_bits(float f) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return __float_as_uint(f);
#elif defined(SYCL_LANGUAGE_VERSION)
  return sycl::bit_cast<uint32_t>(f);
#else
  uint32_t b;
  std::memcpy(&b, &f, sizeof(b));
  return b;
#endif
}

RQ_HD inline float fp8_max(bool fnuz) { return fnuz ? 240.0f : 448.0f; }

RQ_HD inline float fp8_e4m3_to_f32(uint8_t v, bool fnuz) {
  const int bias = fnuz ? 8 : 7;
  const uint32_t sign = (v >> 7) & 1u;
  const uint32_t exp = (v >> 3) & 0xFu;
  const uint32_t mant = v & 0x7u;

  if (fnuz) {
    if (v == 0x80u) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
      return nanf("");
#elif defined(SYCL_LANGUAGE_VERSION)
      return sycl::nan(0u);
#else
      return std::nanf("");
#endif
    }
  } else if (exp == 0xFu && mant == 0x7u) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return nanf("");
#elif defined(SYCL_LANGUAGE_VERSION)
    return sycl::nan(0u);
#else
    return std::nanf("");
#endif
  }

  const float sgn = sign ? -1.0f : 1.0f;
  if (exp == 0u) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return sgn * ldexpf(static_cast<float>(mant), 1 - bias - 3);
#elif defined(SYCL_LANGUAGE_VERSION)
    return sgn * sycl::ldexp(static_cast<float>(mant), 1 - bias - 3);
#else
    return sgn * std::ldexp(static_cast<float>(mant), 1 - bias - 3);
#endif
  }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return sgn * ldexpf(static_cast<float>(8u + mant),
                      static_cast<int>(exp) - bias - 3);
#elif defined(SYCL_LANGUAGE_VERSION)
  return sgn * sycl::ldexp(static_cast<float>(8u + mant),
                           static_cast<int>(exp) - bias - 3);
#else
  return sgn * std::ldexp(static_cast<float>(8u + mant),
                          static_cast<int>(exp) - bias - 3);
#endif
}

RQ_HD inline uint32_t round_to_even(float x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  const float fl = floorf(x);
#elif defined(SYCL_LANGUAGE_VERSION)
  const float fl = sycl::floor(x);
#else
  const float fl = std::floor(x);
#endif
  const float diff = x - fl;
  const uint32_t i = static_cast<uint32_t>(fl);
  if (diff < 0.5f)
    return i;
  if (diff > 0.5f)
    return i + 1u;
  return (i & 1u) ? i + 1u : i;
}

RQ_HD inline uint8_t f32_to_fp8_e4m3(float f, bool fnuz) {
  const int bias = fnuz ? 8 : 7;
  const uint32_t sign = (f32_to_bits(f) >> 31) & 1u;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  const float af = fabsf(f);
  const bool nan_or_inf = isnan(f) || isinf(f);
#elif defined(SYCL_LANGUAGE_VERSION)
  const float af = sycl::fabs(f);
  const bool nan_or_inf = sycl::isnan(f) || sycl::isinf(f);
#else
  const float af = std::fabs(f);
  const bool nan_or_inf = std::isnan(f) || std::isinf(f);
#endif

  if (nan_or_inf) {
    if (fnuz)
      return 0x80u;
    return static_cast<uint8_t>((sign << 7) | 0x7Fu);
  }
  if (af == 0.0f)
    return fnuz ? 0x00u : static_cast<uint8_t>(sign << 7);

  const float max_normal = fp8_max(fnuz);

  int e;
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  const float m = frexpf(af, &e);
#elif defined(SYCL_LANGUAGE_VERSION)
  const float m = sycl::frexp(af, &e);
#else
  const float m = std::frexp(af, &e);
#endif
  int biased = (e - 1) + bias;
  const float frac = m * 2.0f;

  if (biased <= 0) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    const float step = ldexpf(1.0f, 1 - bias - 3);
#elif defined(SYCL_LANGUAGE_VERSION)
    const float step = sycl::ldexp(1.0f, 1 - bias - 3);
#else
    const float step = std::ldexp(1.0f, 1 - bias - 3);
#endif
    uint32_t q = round_to_even(af / step);
    if (q == 0u)
      return fnuz ? 0x00u : static_cast<uint8_t>(sign << 7);
    if (q >= 8u)
      return static_cast<uint8_t>((sign << 7) | (1u << 3) | ((q - 8u) & 7u));
    return static_cast<uint8_t>((sign << 7) | q);
  }

  uint32_t q = round_to_even((frac - 1.0f) * 8.0f);
  if (q == 8u) {
    q = 0u;
    biased += 1;
  }

  if (fnuz) {
    if (biased > 15 || af > max_normal)
      return static_cast<uint8_t>((sign << 7) | 0x7Fu);
  } else {
    if (biased > 15 || (biased == 15 && q >= 7u) || af > max_normal)
      return static_cast<uint8_t>((sign << 7) | 0x7Eu);
  }

  return static_cast<uint8_t>((sign << 7) | (static_cast<uint32_t>(biased) << 3) |
                              (q & 7u));
}

RQ_HD inline uint8_t scaled_fp8(float val, float scale, bool invert, bool fnuz) {
  float x = invert ? val * scale : val / scale;
  const float qmax = fp8_max(fnuz);
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  x = fmaxf(-qmax, fminf(x, qmax));
#elif defined(SYCL_LANGUAGE_VERSION)
  x = sycl::fmax(-qmax, sycl::fmin(x, qmax));
#else
  x = std::fmax(-qmax, std::fmin(x, qmax));
#endif
  return f32_to_fp8_e4m3(x, fnuz);
}

} // namespace rq

#endif
