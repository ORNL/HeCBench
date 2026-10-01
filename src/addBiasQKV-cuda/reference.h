// Portable host reference for the fused add-bias-QKV + transpose kernel.
//
// The device kernel performs, for the QKV tensor laid out as
// [valid_word_num, 3, head, size]:
//   1. decode FP8 (E4M3) source to bfloat16
//   2. out = ((src * input_scale) + bias) * output_scale   (bfloat16 math)
//   3. encode the result back to FP8 (E4M3)
//   4. transpose from [valid_word_num, 3, head, size]
//                  -> [valid_word_num, head, 3, size]
//
// FP8 values are passed as raw bytes (uint8_t) and bfloat16 values as raw 16-bit words (uint16_t).
//
// Two FP8 E4M3 encodings are supported and selected via the `fnuz` flag.

#ifndef ADDBIASQKV_REFERENCE_H
#define ADDBIASQKV_REFERENCE_H

#include <cstdint>
#include <cstring>
#include <cmath>

namespace ref {

inline float bits_to_f32(uint32_t b) {
  float f;
  std::memcpy(&f, &b, sizeof(f));
  return f;
}

inline uint32_t f32_to_bits(float f) {
  uint32_t b;
  std::memcpy(&b, &f, sizeof(b));
  return b;
}

// ---------------------------------------------------------------------------
// bfloat16 (truncated fp32, round-to-nearest-even)
// ---------------------------------------------------------------------------
inline float bf16_to_f32(uint16_t h) {
  return bits_to_f32(static_cast<uint32_t>(h) << 16);
}

inline uint16_t f32_to_bf16(float f) {
  uint32_t x = f32_to_bits(f);
  // Preserve NaN payloads.
  if (((x >> 23) & 0xFF) == 0xFF && (x & 0x7FFFFF) != 0)
    return static_cast<uint16_t>((x >> 16) | 0x0040u);
  uint32_t rounding_bias = ((x >> 16) & 1u) + 0x7FFFu;
  x += rounding_bias;
  return static_cast<uint16_t>(x >> 16);
}

// Emulate a single bfloat16 op: compute in fp32 then round the result to bf16.
inline float bf16_round(float f) { return bf16_to_f32(f32_to_bf16(f)); }

// ---------------------------------------------------------------------------
// FP8 E4M3 decode / encode
// ---------------------------------------------------------------------------
inline float fp8_e4m3_to_f32(uint8_t v, bool fnuz) {
  const int bias = fnuz ? 8 : 7;
  const uint32_t sign = (v >> 7) & 1u;
  const uint32_t exp  = (v >> 3) & 0xFu;
  const uint32_t mant = v & 0x7u;

  if (fnuz) {
    if (v == 0x80u) return std::nanf("");
  } else {
    if (exp == 0xFu && mant == 0x7u) return std::nanf("");
  }

  const float sgn = sign ? -1.0f : 1.0f;
  if (exp == 0u) {
    // subnormal: mant * 2^(1 - bias - 3)
    return sgn * std::ldexp(static_cast<float>(mant), 1 - bias - 3);
  }
  // normal: (8 + mant) * 2^(exp - bias - 3)
  return sgn * std::ldexp(static_cast<float>(8u + mant),
                          static_cast<int>(exp) - bias - 3);
}

inline uint32_t round_to_even(float x) {
  float fl = std::floor(x);
  float diff = x - fl;
  uint32_t i = static_cast<uint32_t>(fl);
  if (diff < 0.5f) return i;
  if (diff > 0.5f) return i + 1u;
  return (i & 1u) ? i + 1u : i;  // tie -> round to even
}

inline uint8_t f32_to_fp8_e4m3(float f, bool fnuz) {
  const int bias = fnuz ? 8 : 7;
  const uint32_t sign = (f32_to_bits(f) >> 31) & 1u;
  const float af = std::fabs(f);

  if (std::isnan(f) || std::isinf(f)) {
    if (fnuz) return 0x80u;
    return static_cast<uint8_t>((sign << 7) | 0x7Fu);
  }
  if (af == 0.0f) return fnuz ? 0x00u : static_cast<uint8_t>(sign << 7);

  const float max_normal = fnuz ? 240.0f : 448.0f;

  int e;
  float m = std::frexp(af, &e);   // af = m * 2^e, m in [0.5, 1)
  int biased = (e - 1) + bias;    // unbiased exponent = e - 1
  float frac = m * 2.0f;          // in [1, 2)

  if (biased <= 0) {
    // subnormal: value = q * 2^(1 - bias - 3)
    float step = std::ldexp(1.0f, 1 - bias - 3);
    uint32_t q = round_to_even(af / step);
    if (q == 0u) return fnuz ? 0x00u : static_cast<uint8_t>(sign << 7);
    if (q >= 8u) {  // rounded up into the smallest normal
      return static_cast<uint8_t>((sign << 7) | (1u << 3) | ((q - 8u) & 7u));
    }
    return static_cast<uint8_t>((sign << 7) | q);  // exponent field 0
  }

  uint32_t q = round_to_even((frac - 1.0f) * 8.0f);
  if (q == 8u) { q = 0u; biased += 1; }

  // Overflow / saturation.
  if (fnuz) {
    if (biased > 15 || af > max_normal)
      return static_cast<uint8_t>((sign << 7) | 0x7Fu);  // saturate to 240
  } else {
    if (biased > 15 || (biased == 15 && q >= 7u) || af > max_normal)
      return static_cast<uint8_t>((sign << 7) | 0x7Eu);  // saturate to 448
  }

  return static_cast<uint8_t>((sign << 7) | (static_cast<uint32_t>(biased) << 3) |
                              (q & 7u));
}

}  // namespace ref

// Host reference for the fused add-bias-QKV + transpose kernel.
//
// src / out are FP8 bytes; bias is bf16 words. Layouts:
//   src  : [valid_word_num, 3, head, size]
//   bias : [3, head, size]
//   out  : [valid_word_num, head, 3, size]   (transposed)
inline void addBiasQKV_reference(
          uint8_t*  out,
    const uint8_t*  src,
    const uint16_t* bias,
    float           input_scale,
    float           output_scale,
    int             valid_word_num,
    int             head_num,
    int             size_per_head,
    int             hidden_unit,
    bool            fnuz)
{
  const float in_s  = ref::bf16_to_f32(ref::f32_to_bf16(input_scale));
  const float out_s = ref::bf16_to_f32(ref::f32_to_bf16(output_scale));

  #pragma omp parallel for collapse(2)
  for (int w = 0; w < valid_word_num; ++w) {
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < head_num; ++h) {
        for (int i = 0; i < size_per_head; ++i) {
          const int src_idx =
              w * 3 * hidden_unit + c * hidden_unit + h * size_per_head + i;
          const int bias_idx = c * hidden_unit + h * size_per_head + i;
          const int tgt_idx = w * 3 * hidden_unit +
                              h * (3 * size_per_head) + c * size_per_head + i;

          float v = ref::bf16_round(ref::fp8_e4m3_to_f32(src[src_idx], fnuz));
          v = ref::bf16_round(v * in_s);
          v = ref::bf16_round(v + ref::bf16_to_f32(bias[bias_idx]));
          v = ref::bf16_round(v * out_s);
          out[tgt_idx] = ref::f32_to_fp8_e4m3(v, fnuz);
        }
      }
    }
  }
}

#endif  // ADDBIASQKV_REFERENCE_H
