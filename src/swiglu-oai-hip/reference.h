#ifndef SWIGLU_OAI_HIP_REFERENCE_H
#define SWIGLU_OAI_HIP_REFERENCE_H

#include "../swiglu-oai-cuda/reference.h"
#include <hip/hip_fp8.h>

enum HipFp8Format { HIP_FP8_OCP = 0, HIP_FP8_FNUZ };

constexpr float kE4M3FnuzMax = 240.f;
constexpr int kE4M3FnuzMaxExp = 7;

inline bool hip_arch_uses_fnuz(const char* arch) {
  return !strncmp(arch, "gfx940", 6) ||
         !strncmp(arch, "gfx941", 6) ||
         !strncmp(arch, "gfx942", 6);
}

inline const char* hip_fp8_format_name(HipFp8Format format) {
  return format == HIP_FP8_FNUZ ? "fnuz" : "ocp";
}

inline float round_to_e4m3_fnuz(float x) {
  if (!(x == x)) return x;
  const float a = fabsf(x);
  const float sign = (x < 0.f) ? -1.f : 1.f;
  if (a >= kE4M3FnuzMax) return sign * kE4M3FnuzMax;
  if (a == 0.f) return 0.f; // FNUZ has one unsigned zero.

  int e;
  frexpf(a, &e);
  e -= 1;
  if (e < -7) e = -7; // FNUZ E4M3 uses exponent bias 8.
  const float quantum = ldexpf(1.f, e - 3);
  const float n = a / quantum;
  const float lower = floorf(n);
  const float fraction = n - lower;
  float rounded;
  if (fraction > 0.5f)
    rounded = lower + 1.f;
  else if (fraction < 0.5f)
    rounded = lower;
  else
    rounded = (fmodf(lower, 2.f) == 0.f) ? lower : lower + 1.f;

  float result = rounded * quantum;
  if (result > kE4M3FnuzMax) result = kE4M3FnuzMax;
  return result == 0.f ? 0.f : sign * result;
}

inline uint8_t hip_fp8_encode(float x, HipFp8Format format) {
  return format == HIP_FP8_FNUZ ? __hip_fp8_e4m3_fnuz(x).__x
                                : __hip_fp8_e4m3(x).__x;
}

inline float hip_fp8_decode(uint8_t x, HipFp8Format format) {
  if (format == HIP_FP8_FNUZ) {
    __hip_fp8_e4m3_fnuz value;
    value.__x = x;
    return (float)value;
  }
  __hip_fp8_e4m3 value;
  value.__x = x;
  return (float)value;
}

inline void quantize_dequantize_input_hip(
    HipFp8Format format, InDtype dtype, const float* input, int64_t n,
    int64_t row_len, float* output, uint8_t* scales, float* tensor_scale) {
  if (format != HIP_FP8_FNUZ ||
      (dtype != IN_FP8 && dtype != IN_MXFP8)) {
    quantize_dequantize_input(
        dtype, input, n, row_len, output, scales, tensor_scale);
    return;
  }

  *tensor_scale = 1.f;
  if (dtype == IN_FP8) {
    float amax = 0.f;
    for (int64_t i = 0; i < n; ++i)
      amax = fmaxf(amax, fabsf(input[i]));
    const float scale = amax > 0.f ? amax / kE4M3FnuzMax : 1.f;
    *tensor_scale = scale;
    for (int64_t i = 0; i < n; ++i)
      output[i] = round_to_e4m3_fnuz(input[i] / scale) * scale;
    return;
  }

  const int64_t blocks_per_row = mx_blocks_per_row(row_len);
  const int64_t rows = n / row_len;
  for (int64_t row = 0; row < rows; ++row) {
    for (int64_t block = 0; block < blocks_per_row; ++block) {
      const int64_t begin = row * row_len + block * MX_BLOCK;
      const int64_t row_end = (row + 1) * row_len;
      const int64_t end = begin + MX_BLOCK < row_end
                        ? begin + MX_BLOCK : row_end;
      float amax = 0.f;
      for (int64_t i = begin; i < end; ++i)
        amax = fmaxf(amax, fabsf(input[i]));
      const uint8_t scale_byte =
          e8m0_scale_byte(amax, kE4M3FnuzMaxExp);
      scales[row * blocks_per_row + block] = scale_byte;
      const float scale = e8m0_to_float(scale_byte);
      for (int64_t i = begin; i < end; ++i)
        output[i] = round_to_e4m3_fnuz(input[i] / scale) * scale;
    }
  }
}

inline void quantize_dequantize_output_hip(
    HipFp8Format format, OutQuant quant, int rows, int dim,
    float* output, uint8_t* scales) {
  if (format != HIP_FP8_FNUZ || quant != OUT_FP8) {
    quantize_dequantize_output(quant, rows, dim, output, scales);
    return;
  }

  const int64_t n = (int64_t)rows * dim;
  float amax = 0.f;
  for (int64_t i = 0; i < n; ++i)
    amax = fmaxf(amax, fabsf(output[i]));
  const float scale = amax > 0.f ? amax / kE4M3FnuzMax : 1.f;
  for (int64_t i = 0; i < n; ++i)
    output[i] = round_to_e4m3_fnuz(output[i] / scale) * scale;
}

inline bool hip_values_match(
    HipFp8Format format, float device, float reference,
    float reference_prequant, OutQuant quant, float scale, float tolerance) {
  if (format != HIP_FP8_FNUZ || quant != OUT_FP8)
    return values_match(device, reference, reference_prequant,
                        quant, scale, tolerance);

  const float difference = fabsf(device - reference);
  if (difference <= tolerance) return true;

  const float a = fabsf(reference_prequant / scale);
  int exponent = -7;
  if (a > 0.f) {
    frexpf(a, &exponent);
    exponent -= 1;
    if (exponent < -7) exponent = -7;
  }
  const float quantum = ldexpf(1.f, exponent - 3);
  const float n = a / quantum;
  const bool near_boundary =
      fabsf(n - floorf(n) - 0.5f) <= 1e-3f;
  return near_boundary &&
         difference <= tolerance + quantum * scale;
}

inline bool self_test_hip_rounding() {
  if (!self_test_rounding()) return false;
  struct Case { float input; float expected; };
  const Case cases[] = {
      {1.0625f, 1.f},
      {1.1875f, 1.25f},
      {1.09f, 1.125f},
      {1e30f, kE4M3FnuzMax},
      {-0.f, 0.f},
  };
  for (const Case& test : cases) {
    const float result = round_to_e4m3_fnuz(test.input);
    if (result != test.expected || signbit(result) != signbit(test.expected)) {
      fprintf(stderr,
              "HIP reference self-test FAILED: FNUZ round(%g)=%g, expected %g\n",
              test.input, result, test.expected);
      return false;
    }
  }
  return true;
}

#endif
