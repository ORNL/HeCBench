// SwiGLU as used by the OpenAI (gpt-oss) MoE, implementing the two-axis
// precision contract defined in ../swiglu-oai-cuda/reference.h: the gate/up
// tensor is stored in one of fp32/fp16/bf16/fp8/mxfp8/fp4/mxfp4 and is
// dequantized by the kernel, and the activation result is optionally quantized
// to fp8 or mxfp4 in the epilogue.
//
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <chrono>
#include <random>
#include <vector>
#include <sycl/sycl.hpp>
#include "reference.h"

using sycl::ext::oneapi::bfloat16;

// ---------------------------------------------------------------------------
// Device-side format conversions. These mirror the host helpers in reference.h
// step for step so that the two sides make identical rounding decisions.
// ---------------------------------------------------------------------------
namespace dev {

// e8m0: biased power-of-two scale. Mirrors e8m0_to_float().
inline float e8m0_decode(uint8_t b) {
  const uint32_t bits =
      b ? static_cast<uint32_t>(b) << 23 : 0x00400000u;
  return sycl::bit_cast<float>(bits);
}

// Mirrors e8m0_scale_byte().
inline uint8_t e8m0_scale_byte(float amax, int elemMaxExp) {
  if (!(amax > 0.f)) return 127;  // 2^0 for an all-zero block
  int e;
  sycl::frexp(amax, &e);
  e -= 1;  // unbiased exponent of amax
  int s = e - elemMaxExp + 127;
  if (s < 0) s = 0;
  if (s > 254) s = 254;
  return static_cast<uint8_t>(s);
}

// Round half to even, as round_to_e4m3() does. Hardware convert instructions
// break ties this way, so roundf()-style half-away-from-zero would disagree.
inline float rne(float n) {
  const float fl = sycl::floor(n);
  const float frac = n - fl;
  if (frac > 0.5f) return fl + 1.f;
  if (frac < 0.5f) return fl;
  return (sycl::fmod(fl, 2.f) == 0.f) ? fl : fl + 1.f;  // tie -> even
}

// OCP e4m3 (bias 7, no infinities) decode. The encoder below saturates at 448,
// so the 0x7f/0xff NaN pattern is never stored and needs no special case here.
inline float e4m3_to_float(uint8_t v) {
  const int exp = (v >> 3) & 0xf;
  const int mant = v & 0x7;
  const float sgn = (v & 0x80u) ? -1.f : 1.f;
  if (exp == 0)  // subnormal: mant * 2^-9
    return sgn * sycl::ldexp(static_cast<float>(mant), -9);
  return sgn * sycl::ldexp(static_cast<float>(8 + mant), exp - 10);
}

// e2m1 decode. Representable magnitudes: 0, 0.5, 1, 1.5, 2, 3, 4, 6.
inline float e2m1_to_float(uint8_t v) {
  const float lut[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  const float m = lut[v & 0x7];
  return (v & 0x8u) ? -m : m;
}

// Encode to OCP e4m3, rounding exactly as round_to_e4m3() does. A negative zero
// encodes as +0, which is numerically identical.
inline uint8_t f32_to_e4m3(float x) {
  const uint8_t sign = (x < 0.f) ? 0x80u : 0x00u;
  const float a = sycl::fabs(x);
  if (a >= kE4M3Max) return sign | 0x7eu;  // 448, largest finite magnitude
  if (a == 0.f) return sign;

  int e;
  sycl::frexp(a, &e);
  e -= 1;
  const int minNormExp = -6;  // e4m3 bias 7 => smallest normal 2^-6
  if (e < minNormExp) e = minNormExp;
  int r = static_cast<int>(
      rne(a / sycl::ldexp(1.f, e - 3)));  // 3 mantissa bits
  if (r >= 16) { r = 8; e += 1; }
  if (e == minNormExp && r < 8)
    return sign | static_cast<uint8_t>(r);  // subnormal
  return sign | static_cast<uint8_t>(((e + 7) << 3) | (r - 8));
}

// Encode to e2m1, mirroring round_to_e2m1() including its round-to-even tie
// rule -- note the strict `> 5.f`, which lets the 4/6 tie at exactly 5 fall
// through to the search and land on 4 (even mantissa).
inline uint8_t f32_to_e2m1(float x) {
  const float lut[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  const uint8_t sign = (x < 0.f) ? 0x8u : 0x0u;
  const float a = sycl::fabs(x);
  if (a > 5.f) return sign | 7u;
  int best = 0;
  float bestd = sycl::fabs(a - lut[0]);
  for (int i = 1; i < 8; i++) {
    const float d = sycl::fabs(a - lut[i]);
    if (d < bestd || (d == bestd && (i % 2) == 0)) {
      bestd = d;
      best = i;
    }
  }
  return sign | static_cast<uint8_t>(best);
}

inline float dev_sigmoid(const float x) {
  if (x >= 0.f) {
    return 1.f / (1.f + sycl::exp(-x));
  } else {
    const float exp_x = sycl::exp(x);
    return exp_x / (1.f + exp_x);
  }
}

}  // namespace dev

template <int STORE>
inline float load_x_raw(const void *X, int64_t index) {
  if constexpr (STORE == XS_F32) {
    return static_cast<const float *>(X)[index];
  } else if constexpr (STORE == XS_F16) {
    return static_cast<float>(static_cast<const sycl::half *>(X)[index]);
  } else if constexpr (STORE == XS_BF16) {
    return static_cast<float>(static_cast<const bfloat16 *>(X)[index]);
  } else if constexpr (STORE == XS_FP8) {
    return dev::e4m3_to_float(static_cast<const uint8_t *>(X)[index]);
  } else {
    const uint8_t byte = static_cast<const uint8_t *>(X)[index >> 1];
    return dev::e2m1_to_float(
        (index & 1) ? static_cast<uint8_t>(byte >> 4)
                    : static_cast<uint8_t>(byte & 0xf));
  }
}

template <int STORE, int SCALE>
inline sycl::float2 load_x_pair(
    const void *X, const uint8_t *Xs, float tensor_scale,
    int64_t index, int64_t scale_index) {
  sycl::float2 pair(
      load_x_raw<STORE>(X, index), load_x_raw<STORE>(X, index + 1));
  float scale = 1.f;
  if constexpr (SCALE == XSC_TENSOR) scale = tensor_scale;
  else if constexpr (SCALE == XSC_MX)
    scale = dev::e8m0_decode(Xs[scale_index]);
  pair *= scale;
  return pair;
}

// The activation for one output element, computed in fp32 whatever the storage
// format is. Gate and linear are interleaved along the last dimension.
template <int BIAS>
inline float load_bias(const KernelArgs &a, int64_t d) {
  if constexpr (BIAS == BS_NONE) {
    (void)a;
    (void)d;
    return 0.f;
  } else if constexpr (BIAS == BS_F16) {
    return static_cast<float>(static_cast<const sycl::half *>(a.B)[d]);
  } else if constexpr (BIAS == BS_BF16) {
    return static_cast<float>(static_cast<const bfloat16 *>(a.B)[d]);
  } else {
    return static_cast<const float *>(a.B)[d];
  }
}

template <int STORE, int SCALE, int BIAS>
inline float swiglu_elem(const KernelArgs &a, int64_t row, int64_t d) {
  const int64_t index = row * static_cast<int64_t>(a.dim) + d;
  const int64_t x_offset = 2 * index;
  int64_t scale_index = 0;
  if constexpr (SCALE == XSC_MX) {
    scale_index = (a.dim % (MX_BLOCK / 2) == 0)
                ? index / (MX_BLOCK / 2)
                : row * a.xBlocksPerRow + (2 * d) / MX_BLOCK;
  }
  const sycl::float2 pair =
      load_x_pair<STORE, SCALE>(a.X, a.Xs, a.xScale, x_offset, scale_index);
  float gate = pair.x() + load_bias<BIAS>(a, 2 * d);
  float linear = pair.y() + load_bias<BIAS>(a, 2 * d + 1);

  gate = sycl::fmin(gate, a.limit);
  linear = sycl::fmin(sycl::fmax(linear, -a.limit), a.limit);

  return gate * dev::dev_sigmoid(a.alpha * gate) * (linear + 1.f);
}

template <int STORE, int SCALE, OutQuant OUT, int BIAS>
void swiglu_oai_kernel(sycl::nd_item<1> item, const KernelArgs a)
{
  const int64_t tiles_per_row =
      (static_cast<int64_t>(a.dim) + item.get_local_range(0) - 1) /
      item.get_local_range(0);
  const int64_t total_tiles =
      static_cast<int64_t>(a.rows) * tiles_per_row;
  const int64_t group_stride =
      static_cast<int64_t>(item.get_group_range(0));
  for (int64_t tile = static_cast<int64_t>(item.get_group(0));
       tile < total_tiles; tile += group_stride) {
    const int64_t row = tile / tiles_per_row;
    const int64_t d = (tile % tiles_per_row) *
                      static_cast<int64_t>(item.get_local_range(0)) +
                      static_cast<int64_t>(item.get_local_id(0));
    const bool active = d < a.dim;
    const float y =
        active ? swiglu_elem<STORE, SCALE, BIAS>(a, row, d) : 0.f;

    if constexpr (OUT == OUT_NONE) {
      if (active)
        static_cast<float *>(a.Y)[
            row * static_cast<int64_t>(a.dim) + d] = y;
    } else if constexpr (OUT == OUT_FP8) {
      if (active)
        static_cast<uint8_t *>(a.Y)[
            row * static_cast<int64_t>(a.dim) + d] =
            dev::f32_to_e4m3(y / a.yScale);
    } else {
      const sycl::sub_group sg = item.get_sub_group();
      float amax = active ? sycl::fabs(y) : 0.f;
      for (int mask = 1; mask < MX_BLOCK; mask <<= 1)
        amax = sycl::fmax(
            amax, sycl::permute_group_by_xor(sg, amax, mask));
      const uint8_t sb = dev::e8m0_scale_byte(amax, kE2M1MaxExp);
      const float s = dev::e8m0_decode(sb);
      const int lane =
          static_cast<int>(sg.get_local_linear_id()) & (MX_BLOCK - 1);

      if (lane == 0 && active)
        a.Ys[row * a.yBlocksPerRow + d / MX_BLOCK] = sb;

      const float paired = sycl::shift_group_left(sg, y, 1);
      if (active && lane % 2 == 0) {
        const float hi = (d + 1 < a.dim) ? paired : 0.f;
        const uint8_t lower_code = dev::f32_to_e2m1(y / s);
        const uint8_t upper_code = dev::f32_to_e2m1(hi / s);
        static_cast<uint8_t *>(a.Y)[row * a.yRowStrideBytes + (d >> 1)] =
            static_cast<uint8_t>(lower_code | (upper_code << 4));
      }
    }
  }
}

template <int STORE, int SCALE, OutQuant OUT, int BIAS, bool TIMED>
class SwigluOAIKernel;

template <int STORE, int SCALE, OutQuant OUT, int BIAS, bool TIMED>
class SwigluOAIKernelSG32;

// `force_sg32` asks for a sub-group of exactly MX_BLOCK lanes. The MXFP4
// epilogue needs a sub-group of at least that width: the XOR masks stay below
// MX_BLOCK, so on a wider sub-group each aligned MX_BLOCK-lane slice reduces
// its own block and elects its own lane 0, but a narrower one would spread a
// block over several sub-groups that each write a different scale for it.
// Devices offering MX_BLOCK take the annotated kernel; devices whose sizes are
// all wider (wave64, say) take the plain one, and main() rejects the rest.
template <int STORE, int SCALE, OutQuant OUT, int BIAS, bool TIMED>
static void submit_swiglu_kernel(sycl::queue &q, sycl::nd_range<1> launch,
                                 const KernelArgs &a, bool force_sg32) {
  q.submit([&](sycl::handler &cgh) {
    if constexpr (OUT == OUT_MXFP4) {
      if (force_sg32) {
        cgh.parallel_for<SwigluOAIKernelSG32<STORE, SCALE, OUT, BIAS, TIMED>>(
            launch, [=](sycl::nd_item<1> item)
                [[sycl::reqd_sub_group_size(MX_BLOCK)]] {
              swiglu_oai_kernel<STORE, SCALE, OUT, BIAS>(item, a);
            });
        return;
      }
    }
    cgh.parallel_for<SwigluOAIKernel<STORE, SCALE, OUT, BIAS, TIMED>>(
        launch, [=](sycl::nd_item<1> item) {
          swiglu_oai_kernel<STORE, SCALE, OUT, BIAS>(item, a);
        });
  });
}

template <int STORE, int SCALE, int BIAS, bool TIMED>
static void submit_swiglu_out(sycl::queue &q, OutQuant out,
                              sycl::nd_range<1> launch, const KernelArgs &a,
                              bool force_sg32) {
  switch (out) {
    case OUT_NONE:
      submit_swiglu_kernel<STORE, SCALE, OUT_NONE, BIAS, TIMED>(
          q, launch, a, force_sg32); break;
    case OUT_FP8:
      submit_swiglu_kernel<STORE, SCALE, OUT_FP8, BIAS, TIMED>(
          q, launch, a, force_sg32); break;
    case OUT_MXFP4:
      submit_swiglu_kernel<STORE, SCALE, OUT_MXFP4, BIAS, TIMED>(
          q, launch, a, force_sg32); break;
    default: break;
  }
}

template <int BIAS, bool TIMED>
static void submit_swiglu_input(sycl::queue &q, InDtype in, OutQuant out,
                                sycl::nd_range<1> launch,
                                const KernelArgs &a, bool force_sg32) {
  switch (in) {
    case IN_FP32:
      submit_swiglu_out<XS_F32, XSC_NONE, BIAS, TIMED>(
          q, out, launch, a, force_sg32); break;
    case IN_FP16:
      submit_swiglu_out<XS_F16, XSC_NONE, BIAS, TIMED>(
          q, out, launch, a, force_sg32); break;
    case IN_BF16:
      submit_swiglu_out<XS_BF16, XSC_NONE, BIAS, TIMED>(
          q, out, launch, a, force_sg32); break;
    case IN_FP8:
      submit_swiglu_out<XS_FP8, XSC_TENSOR, BIAS, TIMED>(
          q, out, launch, a, force_sg32); break;
    case IN_MXFP8:
      submit_swiglu_out<XS_FP8, XSC_MX, BIAS, TIMED>(
          q, out, launch, a, force_sg32); break;
    case IN_FP4:
      submit_swiglu_out<XS_FP4, XSC_TENSOR, BIAS, TIMED>(
          q, out, launch, a, force_sg32); break;
    case IN_MXFP4:
      submit_swiglu_out<XS_FP4, XSC_MX, BIAS, TIMED>(
          q, out, launch, a, force_sg32); break;
    default: break;
  }
}

template <bool TIMED>
static void submit_swiglu(sycl::queue &q, InDtype in, OutQuant out,
                          BiasStore bias, sycl::nd_range<1> launch,
                          const KernelArgs &a, bool force_sg32) {
#define SUBMIT_BIAS(B)                                                        \
  submit_swiglu_input<B, TIMED>(q, in, out, launch, a, force_sg32)
  switch (bias) {
    case BS_NONE: SUBMIT_BIAS(BS_NONE); break;
    case BS_F16:  SUBMIT_BIAS(BS_F16);  break;
    case BS_BF16: SUBMIT_BIAS(BS_BF16); break;
    case BS_F32:  SUBMIT_BIAS(BS_F32);  break;
    default: break;
  }
#undef SUBMIT_BIAS
}

// ---------------------------------------------------------------------------
// Host-side packing and unpacking. The encoders round the same way reference.h
// does, so the value the kernel decodes is bit-identical to the value
// quantize_dequantize_input() left behind.
// ---------------------------------------------------------------------------

static float host_rne(float n) {  // round half to even, as round_to_e4m3() does
  const float fl = floorf(n);
  const float frac = n - fl;
  if (frac > 0.5f) return fl + 1.f;
  if (frac < 0.5f) return fl;
  return (fmodf(fl, 2.f) == 0.f) ? fl : fl + 1.f;
}

static uint8_t f32_to_e4m3_bits(float v) {
  const uint8_t sign = (v < 0.f) ? 0x80u : 0x00u;
  const float a = fabsf(v);
  if (a >= kE4M3Max) return sign | 0x7eu;  // 448, largest finite magnitude
  if (a == 0.f) return sign;

  int e;
  frexpf(a, &e);
  e -= 1;
  if (e < -6) e = -6;
  int r = static_cast<int>(host_rne(a / ldexpf(1.f, e - 3)));
  if (r >= 16) { r = 8; e += 1; }
  if (e == -6 && r < 8)
    return sign | static_cast<uint8_t>(r);  // subnormal
  return sign | static_cast<uint8_t>(((e + 7) << 3) | (r - 8));
}

static float e4m3_bits_to_float(uint8_t v) {
  const int exp = (v >> 3) & 0xf;
  const int mant = v & 0x7;
  const float sgn = (v & 0x80u) ? -1.f : 1.f;
  if (exp == 0) return sgn * ldexpf(static_cast<float>(mant), -9);
  return sgn * ldexpf(static_cast<float>(8 + mant), exp - 10);
}

static const float kE2M1Lut[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};

static uint8_t f32_to_e2m1_nibble(float v) {
  const uint8_t sign = (v < 0.f) ? 0x8u : 0x0u;
  const float a = fabsf(v);
  if (a > 5.f) return sign | 7u;  // strict: exactly 5 is a tie, handled below
  int best = 0;
  float bestd = fabsf(a - kE2M1Lut[0]);
  for (int i = 1; i < 8; i++) {
    const float d = fabsf(a - kE2M1Lut[i]);
    if (d < bestd || (d == bestd && (i % 2) == 0)) {
      bestd = d;
      best = i;
    }
  }
  return sign | static_cast<uint8_t>(best);
}

static float e2m1_nibble_to_float(uint8_t v) {
  const float m = kE2M1Lut[v & 0x7];
  return (v & 0x8u) ? -m : m;
}

// `Xdq` holds the dequantized values the device must reproduce; dividing by the
// element's scale recovers the code that quantize_dequantize_input() chose.
// MX scales are [row][block] over rows of `rowLen` elements.
static void pack_input(InDtype dtype, const float *Xdq, int64_t nelems,
                       int64_t rowLen, int64_t bpr, const uint8_t *scales,
                       float tensor_scale, void *store)
{
  const XStore xstore = x_store_of(dtype);
  const XScale scaling = x_scale_of(dtype);

  auto scale_at = [&](int64_t i) -> float {
    if (scaling == XSC_MX) {
      const int64_t row = i / rowLen;
      const int64_t col = i - row * rowLen;
      return e8m0_to_float(scales[row * bpr + col / MX_BLOCK]);
    }
    return tensor_scale;
  };

  if (xstore == XS_FP4) {
    uint8_t *p = static_cast<uint8_t *>(store);
    for (int64_t i = 0; i < nelems; i += 2) {
      const float scale = scale_at(i);
      const uint8_t lo = f32_to_e2m1_nibble(Xdq[i] / scale);
      const uint8_t hi = (i + 1 < nelems)
          ? f32_to_e2m1_nibble(Xdq[i + 1] / scale) : 0;
      p[i >> 1] = static_cast<uint8_t>(lo | (hi << 4));
    }
    return;
  }

  for (int64_t i = 0; i < nelems; ++i) {
    switch (xstore) {
      case XS_F32:
        static_cast<float *>(store)[i] = Xdq[i];
        break;
      case XS_F16:
        static_cast<sycl::half *>(store)[i] =
            static_cast<sycl::half>(Xdq[i]);
        break;
      case XS_BF16:
        static_cast<bfloat16 *>(store)[i] = static_cast<bfloat16>(Xdq[i]);
        break;
      default:
        static_cast<uint8_t *>(store)[i] =
            f32_to_e4m3_bits(Xdq[i] / scale_at(i));
        break;
    }
  }
}

static int bias_elem_bytes(BiasStore bias) {
  if (bias == BS_F16 || bias == BS_BF16) return 2;
  if (bias == BS_F32) return 4;
  return 0;
}

// Store bias in the requested production type and update the FP32 reference
// values to exactly what the device conversion will produce.
static void pack_bias(BiasStore bias, float *values, int64_t n, void *store) {
  if (bias == BS_NONE) {
    memset(values, 0, static_cast<size_t>(n) * sizeof(float));
    return;
  }
  if (bias == BS_F16) {
    sycl::half *p = static_cast<sycl::half *>(store);
    for (int64_t i = 0; i < n; ++i) {
      p[i] = static_cast<sycl::half>(values[i]);
      values[i] = static_cast<float>(p[i]);
    }
    return;
  }
  if (bias == BS_BF16) {
    bfloat16 *p = static_cast<bfloat16 *>(store);
    for (int64_t i = 0; i < n; ++i) {
      p[i] = static_cast<bfloat16>(values[i]);
      values[i] = static_cast<float>(p[i]);
    }
    return;
  }
  memcpy(store, values, static_cast<size_t>(n) * sizeof(float));
}

// Decodes the quantized device output back to float using the scales the device
// itself produced, so the comparison sees exactly what the kernel wrote.
static void unpack_output(OutQuant q, int rows, int dim, const void *storage,
                          const uint8_t *scales, float tensor_scale, float *Y)
{
  if (q == OUT_NONE) {
    memcpy(Y, storage, static_cast<size_t>(rows) * dim * sizeof(float));
    return;
  }
  const uint8_t *store = static_cast<const uint8_t *>(storage);
  if (q == OUT_FP8) {
    const int64_t n = static_cast<int64_t>(rows) * dim;
    for (int64_t i = 0; i < n; i++)
      Y[i] = e4m3_bits_to_float(store[i]) * tensor_scale;
    return;
  }

  // OUT_MXFP4
  const int64_t bpr = mx_blocks_per_row(dim);
  const int64_t rowBytes = mxfp4_row_bytes(dim);
  for (int64_t r = 0; r < rows; r++) {
    for (int64_t b = 0; b < bpr; b++) {
      const int64_t beg = b * MX_BLOCK;
      const int64_t n = ((beg + MX_BLOCK < dim) ? beg + MX_BLOCK : dim) - beg;
      const float s = e8m0_to_float(scales[r * bpr + b]);
      const uint8_t *src = store + r * rowBytes + b * (MX_BLOCK / 2);
      for (int64_t t = 0; t < n; t++) {
        const uint8_t nibble = (t & 1) ? (src[t >> 1] >> 4) : (src[t >> 1] & 0xf);
        Y[r * dim + beg + t] = e2m1_nibble_to_float(nibble) * s;
      }
    }
  }
}

int main(int argc, char* argv[])
{
  if (argc < 4 || argc > 7) {
    printf("Usage: %s <number of rows> <last dimension> <repeat> "
           "[in_dtype] [out_quant] [bias_dtype]\n", argv[0]);
    printf("  in_dtype : fp32 (default), fp16, bf16, fp8, mxfp8, fp4, mxfp4\n");
    printf("  out_quant: none (default), fp8, mxfp4\n");
    printf("  bias_dtype: none (default), fp16, bf16, fp32 "
           "(fp32 is accuracy-only)\n");
    return 1;
  }

  const int rows = atoi(argv[1]);
  const int dim = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  if (rows <= 0 || dim <= 0) {
    fprintf(stderr, "Error: rows and dimension must be positive (got %d and %d)\n",
            rows, dim);
    return 1;
  }

  // The average kernel time is divided by the repeat count.
  if (repeat <= 0) {
    fprintf(stderr, "Error: repeat count must be positive (got %d)\n", repeat);
    return 1;
  }

  // Assert the rounding rules directly. The end-to-end comparison cannot catch
  // a tie-policy bug -- a tie sits exactly on a rounding boundary, where
  // values_match() grants slack by design -- so this is what guards it.
  if (!self_test_rounding()) return 1;

  const InDtype in_dtype = (argc > 4) ? parse_in_dtype(argv[4]) : IN_FP32;
  if (in_dtype == IN_INVALID) {
    fprintf(stderr, "Error: unknown input dtype '%s' "
            "(valid: fp32 fp16 bf16 fp8 mxfp8 fp4 mxfp4)\n", argv[4]);
    return 1;
  }

  const OutQuant out_quant = (argc > 5) ? parse_out_quant(argv[5]) : OUT_NONE;
  if (out_quant == OUT_INVALID) {
    fprintf(stderr, "Error: unknown output quantization '%s' "
            "(valid: none fp8 mxfp4)\n", argv[5]);
    return 1;
  }

  const BiasStore bias =
      (argc > 6) ? parse_bias_store(argv[6]) : BS_NONE;
  if (static_cast<int>(bias) < static_cast<int>(BS_NONE) || bias > BS_F32) {
    fprintf(stderr, "Error: invalid bias dtype '%s' "
            "(valid: none fp16 bf16 fp32)\n", argv[6]);
    return 1;
  }

  printf("Shape of input tensor: ( %d %d )  in=%s out=%s bias=%s\n",
         rows, 2 * dim, in_dtype_name(in_dtype), out_quant_name(out_quant),
         bias_store_name(bias));

  const int64_t x_row_len = static_cast<int64_t>(2) * dim;
  const int64_t x_nelems = static_cast<int64_t>(rows) * x_row_len;
  const int64_t y_nelems = static_cast<int64_t>(rows) * dim;
  const int64_t b_nelems = x_row_len;

  const size_t x_bytes = x_nelems * sizeof(float);
  const size_t y_bytes = y_nelems * sizeof(float);
  const size_t b_bytes = b_nelems * sizeof(float);
  const size_t b_store_bytes =
      static_cast<size_t>(b_nelems) *
      static_cast<size_t>(bias_elem_bytes(bias));

  // Bytes of the storage representation the kernel actually reads. fp4 packs
  // two elements per byte.
  const size_t x_store_bytes =
      static_cast<size_t>(
          (x_nelems * in_elem_bits(in_dtype) + 7) / 8);

  // MX blocks are cut inside a row on both axes; the input row is 2*dim long.
  const int64_t x_blocks_per_row = mx_blocks_per_row(x_row_len);
  const int64_t y_blocks_per_row = mx_blocks_per_row(dim);
  const int64_t x_nscales =
      in_is_mx(in_dtype)
          ? static_cast<int64_t>(rows) * x_blocks_per_row : 0;
  const int64_t y_nscales =
      out_is_mx(out_quant)
          ? static_cast<int64_t>(rows) * y_blocks_per_row : 0;
  const int64_t y_row_bytes = mxfp4_row_bytes(dim);

  // fp8 stores one byte per element; mxfp4 pads each row to whole MX blocks.
  const size_t y_store_bytes = (out_quant == OUT_NONE) ? y_bytes
      : (out_quant == OUT_FP8) ? static_cast<size_t>(y_nelems)
          : static_cast<size_t>(
                static_cast<int64_t>(rows) * y_row_bytes);

  float *X = static_cast<float*>(malloc(x_bytes));
  float *X_dq = static_cast<float*>(malloc(x_bytes));
  float *B = static_cast<float*>(malloc(b_bytes));
  float *Y = static_cast<float*>(malloc(y_bytes));
  float *Y_ref = static_cast<float*>(malloc(y_bytes));
  // values_match() needs the reference from before the output quant: once
  // quantized every reference value sits exactly on a code, which says nothing
  // about whether a ULP-level difference could have flipped it.
  float *Y_pre = static_cast<float*>(malloc(y_bytes));
  void *X_store = malloc(x_store_bytes);
  void *B_store = (bias == BS_NONE) ? nullptr : malloc(b_store_bytes);
  void *Y_store = malloc(y_store_bytes);
  uint8_t *x_scales =
      static_cast<uint8_t*>(malloc(x_nscales ? x_nscales : 1));
  uint8_t *y_scales =
      static_cast<uint8_t*>(malloc(y_nscales ? y_nscales : 1));
  uint8_t *y_scales_ref =
      static_cast<uint8_t*>(malloc(y_nscales ? y_nscales : 1));

  if (!X || !X_dq || !B || !Y || !Y_ref || !Y_pre || !X_store ||
      (bias != BS_NONE && !B_store) || !Y_store || !x_scales ||
      !y_scales || !y_scales_ref) {
    fprintf(stderr, "Error: host allocation failed\n");
    return EXIT_FAILURE;
  }

  std::default_random_engine generator(123);
  // The range straddles the clamp limit so both clamped and unclamped
  // elements are exercised.
  std::uniform_real_distribution<float> distribution(-12.f, 12.f);

  for (int64_t i = 0; i < x_nelems; i++) {
    X[i] = distribution(generator);
  }
  for (int64_t i = 0; i < b_nelems; i++) {
    B[i] = distribution(generator);
  }
  pack_bias(bias, B, b_nelems, B_store);

  // The exact float values the device must see once it has dequantized.
  float x_scale = 1.f;
  quantize_dequantize_input(in_dtype, X, x_nelems, x_row_len, X_dq, x_scales,
                            &x_scale);
  pack_input(in_dtype, X_dq, x_nelems, x_row_len, x_blocks_per_row, x_scales,
             x_scale, X_store);

  ComputeSwigluOAI(rows, dim, kSwigluAlpha, kSwigluLimit, X_dq, B, Y_ref);
  memcpy(Y_pre, Y_ref, y_bytes);

  // aiter's fused per-tensor quant scales by a factor calibrated ahead of time
  // rather than reducing over the whole tensor inside the kernel, so the fp8
  // output scale is derived here (from the amax quantize_dequantize_output()
  // itself uses) and passed in.
  float y_scale = 1.f;
  if (out_quant == OUT_FP8) {
    float amax = 0.f;
    for (int64_t i = 0; i < y_nelems; i++) {
      const float a = fabsf(Y_ref[i]);
      if (a > amax) amax = a;
    }
    y_scale = (amax > 0.f) ? (amax / kE4M3Max) : 1.f;
  }

  quantize_dequantize_output(out_quant, rows, dim, Y_ref, y_scales_ref);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // The MXFP4 epilogue reduces an MX block within one sub-group, so every
  // sub-group the device may hand the kernel has to be at least MX_BLOCK
  // lanes wide. When MX_BLOCK itself is available the kernel asks for exactly
  // that; otherwise the device only offers wider sub-groups and the plain
  // kernel is safe. A device that can pick a narrower size cannot run the
  // path correctly, so it is skipped rather than silently mis-quantized.
  bool force_sg32 = false;
  if (out_quant == OUT_MXFP4) {
    const std::vector<size_t> sg_sizes =
        q.get_device().get_info<sycl::info::device::sub_group_sizes>();
    force_sg32 = std::find(sg_sizes.begin(), sg_sizes.end(),
                           static_cast<size_t>(MX_BLOCK)) != sg_sizes.end();
    const size_t smallest = sg_sizes.empty()
        ? 0 : *std::min_element(sg_sizes.begin(), sg_sizes.end());
    if (!force_sg32 && smallest < static_cast<size_t>(MX_BLOCK)) {
      printf("Skipped: mxfp4 output needs a sub-group of at least %d lanes, "
             "which %s does not guarantee\n", MX_BLOCK,
             q.get_device().get_info<sycl::info::device::name>().c_str());
      return EXIT_SUCCESS;
    }
  }

  void *d_X = sycl::malloc_device(x_store_bytes, q);
  void *d_B = (bias == BS_NONE)
      ? nullptr : sycl::malloc_device(b_store_bytes, q);
  void *d_Y = sycl::malloc_device(y_store_bytes, q);
  uint8_t *d_Xscales =
      sycl::malloc_device<uint8_t>(x_nscales ? x_nscales : 1, q);
  uint8_t *d_Yscales =
      sycl::malloc_device<uint8_t>(y_nscales ? y_nscales : 1, q);

  if (!d_X || (bias != BS_NONE && !d_B) || !d_Y ||
      !d_Xscales || !d_Yscales) {
    fprintf(stderr, "Error: device allocation failed\n");
    return EXIT_FAILURE;
  }

  q.memcpy(d_X, X_store, x_store_bytes);
  if (d_B) q.memcpy(d_B, B_store, b_store_bytes);
  if (x_nscales) q.memcpy(d_Xscales, x_scales, x_nscales);

  KernelArgs args;
  args.rows = rows;
  args.dim = dim;
  args.alpha = kSwigluAlpha;
  args.limit = kSwigluLimit;
  args.X = d_X;
  args.Xs = d_Xscales;
  args.xScale = x_scale;
  args.B = d_B;
  args.Y = d_Y;
  args.Ys = d_Yscales;
  args.yScale = y_scale;
  args.xBlocksPerRow = x_blocks_per_row;
  args.yRowStrideBytes = y_row_bytes;
  args.yBlocksPerRow = y_blocks_per_row;
  const int block_size = 256;
  const int64_t tiles_per_row = (dim + block_size - 1) / block_size;
  const int64_t total_tiles = static_cast<int64_t>(rows) * tiles_per_row;
  const int64_t max_groups = 1 << 20;
  const int64_t groups =
      (total_tiles < max_groups) ? total_tiles : max_groups;
  sycl::range<1> gws(static_cast<size_t>(groups) * block_size);
  sycl::range<1> lws (block_size);
  sycl::nd_range<1> launch(gws, lws);

  // check correctness before benchmarking
  submit_swiglu<false>(q, in_dtype, out_quant, bias, launch, args,
                       force_sg32);

  q.memcpy(Y_store, d_Y, y_store_bytes);
  if (y_nscales) q.memcpy(y_scales, d_Yscales, y_nscales);
  q.wait_and_throw();
  unpack_output(out_quant, rows, dim, Y_store, y_scales, y_scale, Y);

  const float tol = 1e-3f;

  bool ok = true;
  float maxdiff = 0.f;
  int64_t worst = 0;
  for (int64_t i = 0; i < y_nelems; i++) {
    // The block scale sets the size of one quantization step for this element.
    const float scale = out_is_mx(out_quant)
        ? e8m0_to_float(y_scales_ref[(i / dim) * y_blocks_per_row +
                                     (i % dim) / MX_BLOCK])
        : y_scale;
    const float diff = fabsf(Y[i] - Y_ref[i]);
    if (diff > maxdiff) {
      maxdiff = diff;
      worst = i;
    }
    if (!values_match(Y[i], Y_ref[i], Y_pre[i], out_quant, scale, tol))
      ok = false;
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) {
    int64_t scale_mismatch = 0;
    for (int64_t i = 0; i < y_nscales; ++i)
      if (y_scales[i] != y_scales_ref[i]) ++scale_mismatch;
    fprintf(stderr, "  max |device - reference| = %g at %ld "
            "(device %g, reference %g, tol %g)\n",
            maxdiff, static_cast<long>(worst), Y[worst], Y_ref[worst], tol);
    if (y_nscales)
      fprintf(stderr, "  mismatching e8m0 output scales: %ld of %ld\n",
              static_cast<long>(scale_mismatch),
              static_cast<long>(y_nscales));
  }

  if (bias == BS_F32) {
    printf("FP32 bias is accuracy-only; performance timing skipped.\n");
  } else {
    q.wait_and_throw();
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; i++)
      submit_swiglu<true>(q, in_dtype, out_quant, bias, launch, args,
                          force_sg32);
    q.wait_and_throw();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    end - start).count();
    printf("Average execution time of SwiGLU (OAI) kernel: %f (us)\n",
           (time * 1e-3f) / repeat);
  }

  free(X);
  free(X_dq);
  free(B);
  free(Y);
  free(Y_ref);
  free(Y_pre);
  free(X_store);
  free(B_store);
  free(Y_store);
  free(x_scales);
  free(y_scales);
  free(y_scales_ref);
  sycl::free(d_X, q);
  if (d_B) sycl::free(d_B, q);
  sycl::free(d_Y, q);
  sycl::free(d_Xscales, q);
  sycl::free(d_Yscales, q);

  return 0;
}
