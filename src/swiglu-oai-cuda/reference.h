// The gate and linear halves are interleaved along the last dimension, as they
// are in gpt-oss: element 2*d is the gate, element 2*d+1 is the linear term.
// Both come out of a fused up-projection, so the bias is added here
//
//   gate   = min(x[2*d]     + b[2*d],     limit)
//   linear = clamp(x[2*d+1] + b[2*d+1], -limit, limit)
//   y[d]   = gate * sigmoid(alpha * gate) * (linear + 1)
//
// Note the asymmetry: the gate is clamped from above only, the linear term
// from both sides. That is the reference behaviour, not an oversight. It also
// means `gate` is unbounded below, which is why the sigmoid below is the
// numerically stable form -- and why device kernels must use the stable form
// too rather than the naive 1/(1+exp(-x)).
//
// Two independent precision axes are modelled, mirroring aiter's fused MoE
// stage-1 kernel:
//
//   input  -- the gate/up tensor is stored as fp32/fp16/bf16/fp8/mxfp8/
//             fp4/mxfp4 and dequantized before the activation.
//   output -- the activation result is quantized to fp8 or mxfp4 in the
//             kernel epilogue (aiter calls this the fused quant).
//
// The "mx" formats are OCP microscaling: one e8m0 (power-of-two) scale per
// MX_BLOCK elements along the last dimension. The plain fp8/fp4 formats use a
// single per-tensor scale, matching aiter's QuantType::per_Tensor.

#ifndef REFERENCE_H
#define REFERENCE_H

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

constexpr float kSwigluAlpha = 1.702f;
constexpr float kSwigluLimit = 7.f;

// OCP microscaling block length.
constexpr int MX_BLOCK = 32;

enum InDtype {
  IN_FP32 = 0,
  IN_FP16,
  IN_BF16,
  IN_FP8,
  IN_MXFP8,
  IN_FP4,
  IN_MXFP4,
  IN_INVALID
};

enum OutQuant : int { OUT_NONE = 0, OUT_FP8, OUT_MXFP4, OUT_INVALID };

// Physical input storage and its independent scaling policy. Keeping these
// axes separate lets every backend share one load implementation between,
// for example, FP8 and MXFP8.
enum XStore { XS_F32 = 0, XS_F16, XS_BF16, XS_FP8, XS_FP4 };
enum XScale { XSC_NONE = 0, XSC_TENSOR, XSC_MX };

enum BiasStore { BS_NONE = 0, BS_F16, BS_BF16, BS_F32 };

struct KernelArgs {
  int rows;
  int dim;
  float alpha;
  float limit;
  const void* X;       // packed gate/up tensor
  const uint8_t* Xs;   // e8m0 scales [row][block], mx inputs only
  float xScale;        // per-tensor scale, plain fp8/fp4 only
  const void* B;       // optional packed bias
  void* Y;             // packed activation
  uint8_t* Ys;         // e8m0 scales [row][block], mxfp4 output only
  float yScale;        // per-tensor scale, fp8 output only
  int64_t xBlocksPerRow;
  int64_t yRowStrideBytes;
  int64_t yBlocksPerRow;
};

inline InDtype parse_in_dtype(const char* s) {
  if (!strcmp(s, "fp32")) return IN_FP32;
  if (!strcmp(s, "fp16")) return IN_FP16;
  if (!strcmp(s, "bf16")) return IN_BF16;
  if (!strcmp(s, "fp8")) return IN_FP8;
  if (!strcmp(s, "mxfp8")) return IN_MXFP8;
  if (!strcmp(s, "fp4")) return IN_FP4;
  if (!strcmp(s, "mxfp4")) return IN_MXFP4;
  return IN_INVALID;
}

inline OutQuant parse_out_quant(const char* s) {
  if (!strcmp(s, "none")) return OUT_NONE;
  if (!strcmp(s, "fp8")) return OUT_FP8;
  if (!strcmp(s, "mxfp4")) return OUT_MXFP4;
  return OUT_INVALID;
}

inline BiasStore parse_bias_store(const char* s) {
  if (!strcmp(s, "none")) return BS_NONE;
  if (!strcmp(s, "fp16")) return BS_F16;
  if (!strcmp(s, "bf16")) return BS_BF16;
  if (!strcmp(s, "fp32")) return BS_F32;
  return (BiasStore)-1;
}

inline const char* bias_store_name(BiasStore b) {
  switch (b) {
    case BS_NONE: return "none";
    case BS_F16:  return "fp16";
    case BS_BF16: return "bf16";
    case BS_F32:  return "fp32-accuracy";
    default:      return "invalid";
  }
}

inline XStore x_store_of(InDtype t) {
  switch (t) {
    case IN_FP32: return XS_F32;
    case IN_FP16: return XS_F16;
    case IN_BF16: return XS_BF16;
    case IN_FP8:
    case IN_MXFP8: return XS_FP8;
    default: return XS_FP4;
  }
}

inline XScale x_scale_of(InDtype t) {
  if (t == IN_MXFP8 || t == IN_MXFP4) return XSC_MX;
  if (t == IN_FP8 || t == IN_FP4) return XSC_TENSOR;
  return XSC_NONE;
}

inline const char* in_dtype_name(InDtype t) {
  static const char* n[] = {"fp32", "fp16", "bf16", "fp8",
                            "mxfp8", "fp4", "mxfp4", "invalid"};
  return n[t];
}

// MXFP8 ?
inline const char* out_quant_name(OutQuant q) {
  static const char* n[] = {"none", "fp8", "mxfp4", "invalid"};
  return n[q];
}

// True when the format carries one e8m0 scale per MX_BLOCK elements.
inline bool in_is_mx(InDtype t) { return t == IN_MXFP8 || t == IN_MXFP4; }
inline bool out_is_mx(OutQuant q) { return q == OUT_MXFP4; }

// True when the format carries a single per-tensor scale.
inline bool in_is_scaled(InDtype t) { return t == IN_FP8 || t == IN_FP4; }

// Bits of storage per element. fp4 returns 4; callers pack two elements per
// byte. Only IN_INVALID returns 0.
inline int in_elem_bits(InDtype t) {
  switch (t) {
    case IN_FP32: return 32;
    case IN_FP16:
    case IN_BF16: return 16;
    case IN_FP8:
    case IN_MXFP8: return 8;
    case IN_FP4:
    case IN_MXFP4: return 4;
    default: return 0;
  }
}

// ---------------------------------------------------------------------------
// Scalar format conversions, host side. These define ground truth; device
// kernels use the vendor's native types and must agree with these.
// ---------------------------------------------------------------------------

// Largest finite magnitude of each element format, used to derive MX scales.
constexpr float kE4M3Max = 448.f;   // OCP e4m3
constexpr float kE2M1Max = 6.f;     // e2m1
// Exponent of the largest power of two <= the format max.
constexpr int kE4M3MaxExp = 8;      // 2^8 = 256 <= 448
constexpr int kE2M1MaxExp = 2;      // 2^2 = 4   <= 6

inline float round_to_fp16(float x) {
  // Round-to-nearest-even through the 16-bit binary16 encoding.
  uint32_t u;
  memcpy(&u, &x, 4);
  uint32_t sign = (u >> 16) & 0x8000u;
  int32_t exp = (int32_t)((u >> 23) & 0xff) - 127 + 15;
  uint32_t man = u & 0x7fffffu;

  // Infinities and NaNs are representable in binary16, so pass them through
  // unchanged. (Decoding them with the normal-number formula below would be
  // wrong: exponent 0x1f is reserved, and doing so turned +inf into 65536.)
  if (((u >> 23) & 0xff) == 0xff) return x;

  uint32_t h;
  if (exp >= 0x1f) {
    h = sign | 0x7c00u;  // overflow to inf
  } else if (exp <= 0) {
    if (exp < -10) {
      h = sign;  // underflow to zero
    } else {
      man |= 0x800000u;
      uint32_t shift = (uint32_t)(14 - exp);
      uint32_t sub = man >> shift;
      uint32_t rem = man & ((1u << shift) - 1u);
      uint32_t halfv = 1u << (shift - 1);
      if (rem > halfv || (rem == halfv && (sub & 1u))) sub++;
      h = sign | sub;
    }
  } else {
    uint32_t sub = man >> 13;
    uint32_t rem = man & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (sub & 1u))) {
      sub++;
      if (sub == 0x400u) {
        sub = 0;
        exp++;
        if (exp >= 0x1f) return sign ? -INFINITY : INFINITY;
      }
    }
    h = sign | ((uint32_t)exp << 10) | sub;
  }

  // Decode back to float.
  uint32_t s = (h & 0x8000u) << 16;
  uint32_t e = (h >> 10) & 0x1fu;
  uint32_t m = h & 0x3ffu;
  float r;
  if (e == 0) {
    if (m == 0) {
      memcpy(&r, &s, 4);
      return r;
    }
    float v = (float)m * 5.9604644775390625e-8f;  // 2^-24
    r = s ? -v : v;
    return r;
  }
  if (e == 0x1f) return s ? -INFINITY : INFINITY;
  uint32_t o = s | ((e + 112u) << 23) | (m << 13);
  memcpy(&r, &o, 4);
  return r;
}

inline float round_to_bf16(float x) {
  uint32_t u;
  memcpy(&u, &x, 4);
  if (((u >> 23) & 0xff) == 0xff) return x;  // inf/nan pass through
  uint32_t lsb = (u >> 16) & 1u;
  uint32_t rounded = u + 0x7fffu + lsb;      // round to nearest even
  rounded &= 0xffff0000u;
  float r;
  memcpy(&r, &rounded, 4);
  return r;
}

// Round a float to the nearest representable OCP e4m3 value (saturating).
inline float round_to_e4m3(float x) {
  if (!(x == x)) return x;  // NaN
  float a = fabsf(x);
  float sign = (x < 0.f) ? -1.f : 1.f;
  if (a >= kE4M3Max) return sign * kE4M3Max;
  if (a == 0.f) return x;

  int e;
  frexpf(a, &e);  // a in [0.5,1) * 2^e  =>  unbiased exponent is e-1
  e -= 1;
  const int minNormExp = -6;  // e4m3 bias 7 => smallest normal 2^-6
  if (e < minNormExp) e = minNormExp;
  // 3 mantissa bits: quantum is 2^(e-3). Ties go to even, matching OCP and the
  // hardware convert instructions -- roundf() would break them away from zero
  // and disagree with any backend that rounds in hardware.
  float q = ldexpf(1.f, e - 3);
  const float n = a / q;
  const float fl = floorf(n);
  const float frac = n - fl;
  float rn;
  if (frac > 0.5f) rn = fl + 1.f;
  else if (frac < 0.5f) rn = fl;
  else rn = (fmodf(fl, 2.f) == 0.f) ? fl : fl + 1.f;  // tie -> even
  float r = rn * q;
  if (r > kE4M3Max) r = kE4M3Max;
  return sign * r;
}

// Round a float to the nearest representable e2m1 value (saturating).
// Representable magnitudes: 0, 0.5, 1, 1.5, 2, 3, 4, 6.
inline float round_to_e2m1(float x) {
  static const float lut[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  float a = fabsf(x);
  float sign = (x < 0.f) ? -1.f : 1.f;
  // Strictly above the 4/6 midpoint saturates. Exactly 5.0 is a tie and must
  // fall through to the search below, which sends it to 4 (even mantissa) as
  // round-to-nearest-even requires.
  if (a > 5.f) return sign * 6.f;
  int best = 0;
  float bestd = fabsf(a - lut[0]);
  for (int i = 1; i < 8; i++) {
    float d = fabsf(a - lut[i]);
    // Ties round to even index, matching round-to-nearest-even on the
    // mantissa bit.
    if (d < bestd || (d == bestd && (i % 2) == 0)) {
      bestd = d;
      best = i;
    }
  }
  return sign * lut[best];
}

// e8m0: an unsigned power-of-two scale, biased by 127. Returns the biased
// exponent byte for a block whose largest magnitude is amax, given the
// exponent of the element format's maximum.
inline uint8_t e8m0_scale_byte(float amax, int elemMaxExp) {
  if (!(amax > 0.f)) return 127;  // 2^0 for an all-zero block
  int e;
  frexpf(amax, &e);
  e -= 1;  // unbiased exponent of amax
  int s = e - elemMaxExp + 127;
  if (s < 0) s = 0;
  if (s > 254) s = 254;
  return (uint8_t)s;
}

inline float e8m0_to_float(uint8_t b) { return ldexpf(1.f, (int)b - 127); }

// ---------------------------------------------------------------------------
// Input preparation.
//
// Produces the exact float values the device will see after it dequantizes,
// so host and device operate on identical inputs and any mismatch is a real
// kernel bug rather than a rounding difference in the data itself.
//
// `scales` receives one e8m0 byte per MX_BLOCK elements for the mx formats,
// and is left untouched otherwise. `tensor_scale` receives the per-tensor
// scale for the plain fp8/fp4 formats and 1.0f otherwise.
// ---------------------------------------------------------------------------

// Number of MX blocks in one row of `rowLen` elements. Blocks are cut inside a
// row and never straddle two rows, so the last block of a row may be short.
// The input and the output paths use the same rule.
inline int64_t mx_blocks_per_row(int64_t rowLen) {
  return (rowLen + MX_BLOCK - 1) / MX_BLOCK;
}

// Row stride, in bytes, of a packed mxfp4 buffer whose logical row is `dim`
// elements. Rows are padded out to a whole number of MX blocks so a byte is
// never shared between two rows -- that keeps nibble writes race-free and,
// more importantly, makes every backend agree on the layout.
inline int64_t mxfp4_row_bytes(int64_t dim) {
  return mx_blocks_per_row(dim) * (MX_BLOCK / 2);
}

// `rowLen` is the logical row length of `in` (2*dim for the gate/up tensor).
// MX scales are laid out as [row][block].
inline void quantize_dequantize_input(
    InDtype dtype,
    const float* in,
    int64_t nelems,
    int64_t rowLen,
    float* out,
    uint8_t* scales,
    float* tensor_scale)
{
  *tensor_scale = 1.f;

  switch (dtype) {
    case IN_FP32:
      for (int64_t i = 0; i < nelems; i++) out[i] = in[i];
      return;
    case IN_FP16:
      for (int64_t i = 0; i < nelems; i++) out[i] = round_to_fp16(in[i]);
      return;
    case IN_BF16:
      for (int64_t i = 0; i < nelems; i++) out[i] = round_to_bf16(in[i]);
      return;
    case IN_FP8:
    case IN_FP4: {
      const float emax = (dtype == IN_FP8) ? kE4M3Max : kE2M1Max;
      float amax = 0.f;
      for (int64_t i = 0; i < nelems; i++) {
        float a = fabsf(in[i]);
        if (a > amax) amax = a;
      }
      const float s = (amax > 0.f) ? (amax / emax) : 1.f;
      *tensor_scale = s;
      for (int64_t i = 0; i < nelems; i++) {
        float q = (dtype == IN_FP8) ? round_to_e4m3(in[i] / s)
                                    : round_to_e2m1(in[i] / s);
        out[i] = q * s;
      }
      return;
    }
    case IN_MXFP8:
    case IN_MXFP4: {
      const int maxExp = (dtype == IN_MXFP8) ? kE4M3MaxExp : kE2M1MaxExp;
      const int64_t bpr = mx_blocks_per_row(rowLen);
      const int64_t nrows = (rowLen > 0) ? (nelems / rowLen) : 0;
      for (int64_t r = 0; r < nrows; r++) {
        const int64_t rowBeg = r * rowLen;
        for (int64_t b = 0; b < bpr; b++) {
          const int64_t beg = rowBeg + b * MX_BLOCK;
          const int64_t rowEnd = rowBeg + rowLen;
          const int64_t end = (beg + MX_BLOCK < rowEnd) ? beg + MX_BLOCK : rowEnd;
          float amax = 0.f;
          for (int64_t i = beg; i < end; i++) {
            float a = fabsf(in[i]);
            if (a > amax) amax = a;
          }
          const uint8_t sb = e8m0_scale_byte(amax, maxExp);
          scales[r * bpr + b] = sb;
          const float s = e8m0_to_float(sb);
          for (int64_t i = beg; i < end; i++) {
            float q = (dtype == IN_MXFP8) ? round_to_e4m3(in[i] / s)
                                          : round_to_e2m1(in[i] / s);
            out[i] = q * s;
          }
        }
      }
      return;
    }
    default:
      return;
  }
}

// ---------------------------------------------------------------------------
// The activation itself.
// ---------------------------------------------------------------------------

// Numerically stable sigmoid: expf(-x) overflows for large negative x, and the
// gate is clamped only from above, so x here really can be very negative.
inline float sigmoid(const float x) {
  if (x >= 0) {
    return 1.f / (1.f + expf(-x));
  } else {
    const float exp_x = expf(x);
    return exp_x / (1.f + exp_x);
  }
}

// Xdata and Bdata are already-dequantized float values.
inline void ComputeSwigluOAI(
    const int rows,
    const int dim,
    const float alpha,
    const float limit,
    const float* Xdata,
    const float* Bdata,
          float* Ydata)
{
  for (int i = 0; i < rows; ++i) {
    const int64_t xOffset = (int64_t)i * 2 * dim;
    const int64_t yOffset = (int64_t)i * dim;
    for (int d = 0; d < dim; ++d) {
      float gate = Xdata[xOffset + 2 * d] + Bdata[2 * d];
      float linear = Xdata[xOffset + 2 * d + 1] + Bdata[2 * d + 1];

      gate = fminf(gate, limit);
      linear = fminf(fmaxf(linear, -limit), limit);

      Ydata[yOffset + d] = gate * sigmoid(alpha * gate) * (linear + 1.f);
    }
  }
}

// ---------------------------------------------------------------------------
// Output quantization (aiter's fused stage-1 epilogue).
//
// Quantizes in place, leaving dequantized float values behind so the result
// can be compared against the device output elementwise. mxfp4 blocks run
// along the last dimension and never straddle a row.
// ---------------------------------------------------------------------------

inline void quantize_dequantize_output(
    OutQuant q,
    int rows,
    int dim,
    float* Ydata,
    uint8_t* scales)
{
  if (q == OUT_NONE) return;

  if (q == OUT_FP8) {
    // Per-tensor scale, as aiter does for a non-microscaled fused quant.
    const int64_t n = (int64_t)rows * dim;
    float amax = 0.f;
    for (int64_t i = 0; i < n; i++) {
      float a = fabsf(Ydata[i]);
      if (a > amax) amax = a;
    }
    const float s = (amax > 0.f) ? (amax / kE4M3Max) : 1.f;
    for (int64_t i = 0; i < n; i++) Ydata[i] = round_to_e4m3(Ydata[i] / s) * s;
    return;
  }

  // OUT_MXFP4
  const int64_t blocksPerRow = mx_blocks_per_row(dim);
  for (int i = 0; i < rows; i++) {
    for (int64_t b = 0; b < blocksPerRow; b++) {
      const int64_t beg = (int64_t)i * dim + b * MX_BLOCK;
      const int64_t rowEnd = (int64_t)i * dim + dim;
      const int64_t end = (beg + MX_BLOCK < rowEnd) ? beg + MX_BLOCK : rowEnd;
      float amax = 0.f;
      for (int64_t k = beg; k < end; k++) {
        float a = fabsf(Ydata[k]);
        if (a > amax) amax = a;
      }
      const uint8_t sb = e8m0_scale_byte(amax, kE2M1MaxExp);
      scales[(int64_t)i * blocksPerRow + b] = sb;
      const float s = e8m0_to_float(sb);
      for (int64_t k = beg; k < end; k++)
        Ydata[k] = round_to_e2m1(Ydata[k] / s) * s;
    }
  }
}

// Additional slack for a QUANTIZED output element.
//
// Host and device quantize with the same rules, so they normally agree
// exactly. But an element sitting within a ULP of a rounding boundary can be
// pushed across it by a 1-ULP expf difference, and that costs a whole
// quantum -- which at these magnitudes is around 4.0, not 0.005. Comparing
// quantized output against a small absolute tolerance is therefore really an
// exact-match test, and it fails by a huge margin when it fails at all.
//
// Allowing exactly one quantum keeps the check meaningful (a genuinely wrong
// kernel is off by far more than one step, as the injected-fault runs
// confirm) without being one unlucky dataset away from a spurious failure.
// `scale` is the e8m0 block scale for mxfp4, or the per-tensor scale for fp8.
//
// The slack is GATED on the reference value actually sitting near a rounding
// boundary. Granting it unconditionally made the fp8 path blind to uniform
// relative errors below one quantum -- and since e4m3 has 3 mantissa bits, a
// quantum is 6.25%-12.5% of the value, so a 2% or 5% error in the activation
// went undetected (measured on the HIP port; it only began failing at ~7%).
//
// Gating fixes that because the two cases live five orders of magnitude apart:
// a 1-ULP expf difference lands ~1.6e-6 of a quantum from a boundary, while a
// 2% error lands ~0.16 of a quantum away. A threshold of 1e-3 quanta forgives
// the former and catches the latter.
//
// This is why values_match() needs the PRE-quantization reference: once
// quantized, every reference value sits exactly on a representable code and is
// therefore always half a quantum from a boundary, which carries no
// information about whether a flip was plausible.
//
// NOTE -- the naive sigmoid is benign HERE, and no input range can catch it.
// The stable form above is good practice and remains what this contract asks
// for. But for this particular expression the naive 1/(1+expf(-x)) is not
// actually wrong: when -alpha*gate overflows, expf returns +inf and
// 1/(1+inf) is exactly 0 -- a finite zero, never NaN -- and by that point the
// true sigmoid is already denormal-small. Measured divergence in y: exactly 0
// at |gate| = 52, 5.0e-43 at |gate| = 60, and exactly 0 again beyond |gate| =
// 80 where both forms give -0. So widening the data range does NOT expose the
// difference; an earlier version of this comment claimed |gate| > 52 would,
// and that was wrong. The naive form can only be found by reading the code,
// not by any output check this benchmark can perform.
inline float allowed_error(OutQuant q, float ref, float scale) {
  if (q == OUT_NONE) return 0.f;

  if (q == OUT_MXFP4) {
    // e2m1 steps are not uniform: 0.5 below 2, then 1.0, then 2.0 up to the
    // 6 ceiling. Use the LOCAL step -- returning the smallest one would reject
    // a legitimate flip at the 4/6 boundary, which is the very case this slack
    // exists to absorb.
    const float a = fabsf(ref / scale);
    if (a < 2.f) return 0.5f * scale;
    if (a < 4.f) return 1.f * scale;
    return 2.f * scale;
  }

  // e4m3: the step is 2^(e-3) at the exponent of the scaled magnitude.
  const float a = fabsf(ref / scale);
  int e = -6;
  if (a > 0.f) {
    frexpf(a, &e);
    e -= 1;
    if (e < -6) e = -6;
  }
  return ldexpf(1.f, e - 3) * scale;
}

// Direct conformance check on the rounding rules.
//
// The end-to-end output comparison CANNOT catch a tie-policy bug, and no
// choice of threshold would let it: a tie sits at distance 0 from a rounding
// boundary, which is the deepest part of the region where values_match()
// grants slack, so "tie-rule disagreement" and "benign ULP noise at a
// boundary" are the same geometry. Exact ties are also vanishingly rare in
// random data -- measured, about 1 e4m3 tie per 16.7M elements -- so even an
// exact-match comparison would only catch one by luck.
//
// Backends therefore assert the rules directly. This is what guards against a
// port that hands ties to a hardware convert with different semantics.
// Returns true if every rule holds, and prints the first failure otherwise.
inline bool self_test_rounding() {
  struct Case { const char* what; float got; float want; };
  const Case cases[] = {
    // e4m3 ties go to even, not away from zero.
    {"round_to_e4m3(1.0625)", round_to_e4m3(1.0625f), 1.0f},
    {"round_to_e4m3(1.1875)", round_to_e4m3(1.1875f), 1.25f},
    {"round_to_e4m3(1.09)",   round_to_e4m3(1.09f),   1.125f},
    {"round_to_e4m3(1e30)",   round_to_e4m3(1e30f),   kE4M3Max},
    // e2m1 ties go to even. 5.0 is the 4-vs-6 midpoint and must give 4.
    {"round_to_e2m1(5.0)",  round_to_e2m1(5.0f),  4.0f},
    {"round_to_e2m1(3.5)",  round_to_e2m1(3.5f),  4.0f},
    {"round_to_e2m1(2.5)",  round_to_e2m1(2.5f),  2.0f},
    {"round_to_e2m1(0.75)", round_to_e2m1(0.75f), 1.0f},
    {"round_to_e2m1(0.25)", round_to_e2m1(0.25f), 0.0f},
    {"round_to_e2m1(100)",  round_to_e2m1(100.f), 6.0f},
    {"round_to_e2m1(-5.0)", round_to_e2m1(-5.0f), -4.0f},
  };
  for (unsigned i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
    if (cases[i].got != cases[i].want) {
      fprintf(stderr, "reference.h self-test FAILED: %s = %g, expected %g\n",
              cases[i].what, cases[i].got, cases[i].want);
      return false;
    }
  }
  // fp16 must preserve inf and NaN rather than decoding them as normals.
  const float pinf = round_to_fp16(INFINITY);
  const float qnan = round_to_fp16(NAN);
  if (!(pinf > 0.f) || !(pinf == pinf * 2.f) || qnan == qnan) {
    fprintf(stderr, "reference.h self-test FAILED: round_to_fp16 inf/NaN "
                    "(+inf -> %g, NaN -> %g)\n", pinf, qnan);
    return false;
  }
  return true;
}

// Was the pre-quantization reference value close enough to a rounding boundary
// that a ULP-level difference in the activation could plausibly have pushed the
// device to the neighbouring code? Distances are measured in units of the local
// quantum, so the threshold is dimensionless.
inline bool near_rounding_boundary(OutQuant q, float pre, float scale) {
  if (q == OUT_NONE) return false;
  const float kBoundaryQuanta = 1e-3f;
  const float a = fabsf(pre / scale);

  if (q == OUT_MXFP4) {
    // Midpoints between the e2m1 magnitudes {0,.5,1,1.5,2,3,4,6}. The quantum
    // is not uniform, so each boundary carries its own local step.
    static const float bnd[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.f};
    static const float step[7] = {0.5f, 0.5f, 0.5f, 0.5f, 1.f, 1.f, 2.f};
    for (int i = 0; i < 7; i++)
      if (fabsf(a - bnd[i]) <= kBoundaryQuanta * step[i]) return true;
    return false;
  }

  // e4m3: boundaries are the odd half-multiples of the local quantum.
  int e = -6;
  if (a > 0.f) {
    frexpf(a, &e);
    e -= 1;
    if (e < -6) e = -6;
  }
  const float qs = ldexpf(1.f, e - 3);
  const float n = a / qs;
  return fabsf(n - floorf(n) - 0.5f) <= kBoundaryQuanta;
}

// Does a device value match the reference for this element?
//   dev          -- device result, dequantized
//   ref          -- reference result after quantize_dequantize_output
//   ref_prequant -- reference result BEFORE that call (keep a copy)
//   scale        -- e8m0 block scale for mxfp4, per-tensor scale for fp8
inline bool values_match(float dev, float ref, float ref_prequant,
                         OutQuant q, float scale, float tol) {
  const float diff = fabsf(dev - ref);
  if (diff <= tol) return true;
  if (q == OUT_NONE) return false;
  return near_rounding_boundary(q, ref_prequant, scale) &&
         diff <= tol + allowed_error(q, ref, scale);
}

#endif
