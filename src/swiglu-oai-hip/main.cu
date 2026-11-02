#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <chrono>
#include <random>
#include <hip/hip_runtime.h>
#include <hip/hip_version.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp8.h>
#include "reference.h"

#ifndef SWIGLU_HAS_HIP_E8M0_CONVERSION
// __hip_cvt_float_to_e8m0 first shipped in the TheRock 7.11 line. Allow an
// explicit build definition to handle vendor backports or unusual versioning.
#if (HIP_VERSION_MAJOR > 7) || \
    (HIP_VERSION_MAJOR == 7 && HIP_VERSION_MINOR >= 11)
#define SWIGLU_HAS_HIP_E8M0_CONVERSION 1
#else
#define SWIGLU_HAS_HIP_E8M0_CONVERSION 0
#endif
#endif

#if !defined(__HIPCC__)
#error "swiglu-oai-hip must be compiled with hipcc"
#endif

#ifndef SWIGLU_HAS_HIP_FP4
// hip_fp4.h first ships with ROCm 7.0. Older installations keep every other
// format and take the software E2M1 codec below, which makes the same
// rounding decisions as reference.h.
#if HIP_VERSION_MAJOR >= 7
#define SWIGLU_HAS_HIP_FP4 1
#else
#define SWIGLU_HAS_HIP_FP4 0
#endif
#endif

#if SWIGLU_HAS_HIP_FP4
#include <hip/hip_fp4.h>
#endif


#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    const hipError_t error_ = (call);                                          \
    if (error_ != hipSuccess) {                                                \
      fprintf(stderr, "HIP error at %s:%d: %s failed: %s\n",                   \
              __FILE__, __LINE__, #call, hipGetErrorString(error_));           \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

static_assert(sizeof(size_t) >= 8,
              "This benchmark requires 64-bit size_t indexing");

// ---------------------------------------------------------------------------
// Architecture-matched E4M3 storage.
//
// CDNA3 gfx940/941/942 uses its native FNUZ format (bias 8, max 240), and the
// HIP-local reference extends the shared reference with matching FNUZ rules.
// gfx950, gfx1200/1201, and gfx1250 use native OCP E4M3 (bias 7, max 448).
// CDNA1/2 have no native FP8 conversion and use the software OCP codec below.
// Raw byte storage keeps host packing independent from device-only
// architecture macros.
// ---------------------------------------------------------------------------

__host__ __device__ __forceinline__ float bits_to_float(uint32_t u) {
  return __builtin_bit_cast(float, u);
}

__host__ __device__ __forceinline__ uint32_t float_to_bits(float f) {
  return __builtin_bit_cast(uint32_t, f);
}

__host__ __device__ __forceinline__ float ocp_e4m3_decode(uint8_t x) {
#if defined(__HIP_DEVICE_COMPILE__) && \
    (defined(__gfx950__) || defined(__gfx1200__) || \
     defined(__gfx1201__) || defined(__gfx1250__))
  __hip_fp8_e4m3 v;
  v.__x = x;
  return static_cast<float>(v);
#else
  const uint32_t sign = static_cast<uint32_t>(x & 0x80u) << 24;
  const uint32_t mag = x & 0x7fu;
  if (mag == 0) return bits_to_float(sign);
  if (mag < 8) {
    const float value = static_cast<float>(mag) * 0.001953125f; // 2^-9
    return sign ? -value : value;
  }
  if (mag == 0x7f) return NAN;
  const uint32_t exponent = mag >> 3;
  const uint32_t mantissa = mag & 7u;
  return bits_to_float(sign | ((exponent + 120u) << 23) |
                       (mantissa << 20));
#endif
}

__host__ __device__ __forceinline__ uint8_t ocp_e4m3_encode(float x) {
#if defined(__HIP_DEVICE_COMPILE__) && \
    (defined(__gfx950__) || defined(__gfx1200__) || \
     defined(__gfx1201__) || defined(__gfx1250__))
  return __hip_fp8_e4m3(x).__x;
#else
  const uint32_t u = float_to_bits(x);
  const uint8_t sign = static_cast<uint8_t>((u >> 24) & 0x80u);
  const uint32_t fexp = (u >> 23) & 0xffu;
  const float a = fabsf(x);
  if (fexp == 0xffu)
    return (u & 0x7fffffu) ? static_cast<uint8_t>(sign | 0x7fu)
                           : static_cast<uint8_t>(sign | 0x7eu);
  if (a >= 448.f) return static_cast<uint8_t>(sign | 0x7eu);
  if (a < 0.015625f) { // E4M3 subnormals, quantum 2^-9
    const float scaled = a * 512.f;
    uint32_t code = static_cast<uint32_t>(scaled);
    const float fraction = scaled - static_cast<float>(code);
    if (fraction > 0.5f || (fraction == 0.5f && (code & 1u))) ++code;
    return static_cast<uint8_t>(sign | code);
  }

  int exponent = static_cast<int>(fexp) - 127;
  const uint32_t significand = 0x800000u | (u & 0x7fffffu);
  uint32_t rounded = significand >> 20;
  const uint32_t remainder = significand & 0xfffffu;
  if (remainder > 0x80000u ||
      (remainder == 0x80000u && (rounded & 1u))) {
    ++rounded;
    if (rounded == 16u) {
      rounded = 8u;
      ++exponent;
    }
  }
  uint32_t code = static_cast<uint32_t>(exponent + 7) * 8u + (rounded - 8u);
  if (code > 0x7eu) code = 0x7eu;
  return static_cast<uint8_t>(sign | code);
#endif
}

__device__ __forceinline__ float device_e4m3_decode(uint8_t x) {
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
  __hip_fp8_e4m3_fnuz value;
  value.__x = x;
  return static_cast<float>(value);
#else
  return ocp_e4m3_decode(x);
#endif
}

__device__ __forceinline__ uint8_t device_e4m3_encode(float x) {
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
  return __hip_fp8_e4m3_fnuz(x).__x;
#else
  return ocp_e4m3_encode(x);
#endif
}

#if SWIGLU_HAS_HIP_FP4
static_assert(sizeof(__hip_fp4x2_e2m1) == 1, "two fp4 values must pack into one byte");
#endif

// ---------------------------------------------------------------------------
// E2M1 codec. Representable magnitudes: 0, 0.5, 1, 1.5, 2, 3, 4, 6. The
// software path is used when hip_fp4.h is unavailable; its nearest-value
// search with a tie-to-even-index rule mirrors round_to_e2m1() in
// reference.h, including the strict `> 5.f` that lets the 4/6 tie land on 4.
// ---------------------------------------------------------------------------

__host__ __device__ __forceinline__ float e2m1_decode(uint8_t nibble) {
#if SWIGLU_HAS_HIP_FP4
  __hip_fp4_e2m1 v;
  v.__x = static_cast<__hip_fp4_storage_t>(nibble & 0xfu);
  return static_cast<float>(v);
#else
  const float lut[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  const float m = lut[nibble & 0x7u];
  return (nibble & 0x8u) ? -m : m;
#endif
}

// Two E2M1 values packed into one byte, low nibble first.
__host__ __device__ __forceinline__ uint8_t e2m1x2_encode(float lo, float hi) {
#if SWIGLU_HAS_HIP_FP4
  return __hip_fp4x2_e2m1(make_float2(lo, hi)).__x;
#else
  auto encode = [](float x) -> uint8_t {
    const float lut[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    const uint8_t sign = (x < 0.f) ? 0x8u : 0x0u;
    const float a = fabsf(x);
    if (a > 5.f) return sign | 7u;
    int best = 0;
    float bestd = fabsf(a - lut[0]);
    for (int i = 1; i < 8; i++) {
      const float d = fabsf(a - lut[i]);
      if (d < bestd || (d == bestd && (i % 2) == 0)) {
        bestd = d;
        best = i;
      }
    }
    return sign | static_cast<uint8_t>(best);
  };
  return static_cast<uint8_t>(encode(lo) | (encode(hi) << 4));
#endif
}

// ---------------------------------------------------------------------------
// Device-side helpers
// ---------------------------------------------------------------------------

// expf(-alpha*gate) overflows for large negative gate.
__device__ __forceinline__ float dev_sigmoid(float x) {
  if (x >= 0.f) {
    return 1.f / (1.f + expf(-x));
  } else {
    const float ex = expf(x);
    return ex / (1.f + ex);
  }
}

// FP32 reserves exponent 0 for zero/subnormals while E8M0 treats exponent 0 as 2^{-127}
// TODO: b = 0xFF
__device__ __forceinline__ float e8m0_decode(uint8_t b) {
  return bits_to_float(
      b ? static_cast<uint32_t>(b) << 23 : 0x00400000u);
}

// Input load: fetch the native element and undo the quantization scale.

template <int STORE>
__device__ __forceinline__ float load_x_raw(
    const void* __restrict__ X, const int64_t idx) {
  if constexpr (STORE == XS_F32) {
    return static_cast<const float*>(X)[idx];
  } else if constexpr (STORE == XS_F16) {
    return __half2float(static_cast<const __half*>(X)[idx]);
  } else if constexpr (STORE == XS_BF16) {
    return __bfloat162float(static_cast<const __hip_bfloat16*>(X)[idx]);
  } else if constexpr (STORE == XS_FP8) {
    return device_e4m3_decode(static_cast<const uint8_t*>(X)[idx]);
  } else {
    return e2m1_decode(static_cast<uint8_t>(
        (static_cast<const uint8_t*>(X)[idx >> 1] >> ((idx & 1) << 2)) & 0xf));
  }
}

// Load one adjacent gate/linear pair. FP4 stores the pair in one byte and FP8
// stores it in one aligned 16-bit word, so one vector conversion decodes both
// values. Other formats retain their ordinary scalar loads.
template <int STORE, int SCALE>
__device__ __forceinline__ float2 load_x_pair(
    const void* __restrict__ X, const uint8_t* __restrict__ Xs,
    const float tensorScale, const int64_t idx, const int64_t scaleIdx) {
  float2 pair;
  if constexpr (STORE == XS_FP4) {
    const uint8_t packed = static_cast<const uint8_t*>(X)[idx >> 1];
#if SWIGLU_HAS_HIP_FP4
    __hip_fp4x2_e2m1 native;
    native.__x = packed;
    pair = static_cast<float2>(native);
#else
    pair = make_float2(e2m1_decode(packed & 0xfu),
                       e2m1_decode(packed >> 4));
#endif
  } else if constexpr (STORE == XS_FP8) {
#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
    __hip_fp8x2_e4m3_fnuz packed;
#else
    __hip_fp8x2_e4m3 packed;
#endif
    packed.__x = static_cast<const uint16_t*>(X)[idx >> 1];
    pair = static_cast<float2>(packed);
  } else {
    pair = make_float2(
        load_x_raw<STORE>(X, idx), load_x_raw<STORE>(X, idx + 1));
  }
  float scale = 1.f;
  if constexpr (SCALE == XSC_TENSOR) scale = tensorScale;
  else if constexpr (SCALE == XSC_MX)
    scale = e8m0_decode(Xs[scaleIdx]);
  pair.x *= scale;
  pair.y *= scale;
  return pair;
}

// Bias is a compile-time choice. In the BS_NONE case the
// compiler replaces this whole function with 0 and removes the bias pointer
// and global-memory load. FP16/BF16 bias is converted to float before the
// activation math. BS_F32 exists for the accuracy-only test path.
template <int BIAS>
__device__ __forceinline__ float load_bias(const void* __restrict__ B,
                                           const int d) {
  if constexpr (BIAS == BS_NONE) {
    (void)B;
    (void)d;
    return 0.f;
  } else if constexpr (BIAS == BS_F16) {
    return __half2float(static_cast<const __half*>(B)[d]);
  } else if constexpr (BIAS == BS_BF16) {
    return __bfloat162float(static_cast<const __hip_bfloat16*>(B)[d]);
  } else {
    return static_cast<const float*>(B)[d];
  }
}

template <int STORE, int SCALE, int BIAS>
__device__ __forceinline__ float swiglu_elem(
    const KernelArgs& a, const int64_t row, const int64_t d) {
  const int64_t index = row * a.dim + d;
  const int64_t xOffset = 2 * index;
  int64_t scaleIdx = 0;
  if constexpr (SCALE == XSC_MX) {
    scaleIdx = (a.dim % (MX_BLOCK / 2) == 0)
             ? index / (MX_BLOCK / 2)
             : row * a.xBlocksPerRow + (2 * d) / MX_BLOCK;
  }
  const float2 pair =
      load_x_pair<STORE, SCALE>(a.X, a.Xs, a.xScale, xOffset, scaleIdx);
  float gate = pair.x + load_bias<BIAS>(a.B, 2 * d);
  float linear = pair.y + load_bias<BIAS>(a.B, 2 * d + 1);

  gate = fminf(gate, a.limit);
  linear = fminf(fmaxf(linear, -a.limit), a.limit);
  return gate * dev_sigmoid(a.alpha * gate) * (linear + 1.f);
}

// One thread per output element. Threads are tiled (row, block of blockDim.x
// columns) rather than laid out flat so that an MXFP4 block always lands on a
// 32-lane logical subgroup and never straddles a row. Shuffle width=MX_BLOCK
// makes that subgroup one full wave when warpSize is 32 and half a wave when
// warpSize is 64. The tile loop is a grid-stride loop over int64 tile ids, so
// both the index and the guard stay 64-bit safe no matter how large rows*dim
// gets.
template <int STORE, int SCALE, OutQuant OUT, int BIAS>
__global__ void swiglu_oai_kernel(const KernelArgs a) {
  const int64_t tilesPerRow =
      (static_cast<int64_t>(a.dim) + blockDim.x - 1) / blockDim.x;
  const int64_t totalTiles = static_cast<int64_t>(a.rows) * tilesPerRow;

  for (int64_t tile = blockIdx.x; tile < totalTiles; tile += gridDim.x) {
    const int64_t row = tile / tilesPerRow;
    const int64_t d =
        (tile % tilesPerRow) * static_cast<int64_t>(blockDim.x) + threadIdx.x;
    const bool active = (d < static_cast<int64_t>(a.dim));

    const float y =
        active ? swiglu_elem<STORE, SCALE, BIAS>(a, row, d) : 0.f;

    if constexpr (OUT == OUT_NONE) {
      if (active)
        static_cast<float*>(a.Y)[row * static_cast<int64_t>(a.dim) + d] = y;
    } else if constexpr (OUT == OUT_FP8) {
      if (active)
        static_cast<uint8_t*>(a.Y)[
            row * static_cast<int64_t>(a.dim) + d] =
            device_e4m3_encode(y / a.yScale);
    } else {  // OUT_MXFP4
      // amax over the 32 lanes of this mx block; inactive lanes contribute 0.
      float amax = active ? fabsf(y) : 0.f;
      for (int m = 1; m < MX_BLOCK; m <<= 1)
        amax = fmaxf(amax, __shfl_xor(amax, m, MX_BLOCK));

      // Dividing by E2M1's largest power-of-two exponent before conversion
      // chooses the block scale. E8M0 round-toward-zero is floor(log2) for
      // positive values, exactly matching reference.h's scale rule. E8M0 has
      // no zero, so retain byte 127 (scale 1) for an all-zero/NaN block.
#if SWIGLU_HAS_HIP_E8M0_CONVERSION
      const uint8_t sb = (amax > 0.f)
          ? __hip_cvt_float_to_e8m0(
                amax * 0.25f, __HIP_SATFINITE, hipRoundZero)
          : 127;
#else
      // Older HIP headers have no E8M0 conversion API. For a positive normal
      // float, its biased IEEE-754 exponent byte minus kE2M1MaxExp is exactly
      // the desired E8M0 byte; subnormals clamp to byte 0.
      const int scaleByte =
          static_cast<int>((__float_as_uint(amax) >> 23) & 0xffu) -
          kE2M1MaxExp;
      const uint8_t sb = !(amax > 0.f) ? 127
                       : scaleByte < 0 ? 0
                       : scaleByte > 254 ? 254
                                         : static_cast<uint8_t>(scaleByte);
#endif
      const float s = e8m0_decode(sb);

      // Lane 0 of the block owns the scale. If its column is out of range the
      // whole block is, so there is nothing to write.
      if ((threadIdx.x & (MX_BLOCK - 1)) == 0 && active)
        a.Ys[row * a.yBlocksPerRow + (d / MX_BLOCK)] = sb;

      // Two fp4 values share a byte. The even lane collects its neighbour's
      // float and converts the pair together, matching CUDA's packed path.
      const float paired = __shfl_down(y, 1, 2);
      if (active && (d & 1) == 0) {
        const float hi =
            (d + 1 < static_cast<int64_t>(a.dim)) ? paired : 0.f;
        static_cast<uint8_t*>(a.Y)[
            row * a.yRowStrideBytes + (d >> 1)] =
            e2m1x2_encode(y / s, hi / s);
      }
    }
  }
}

template <int STORE, int SCALE, int BIAS>
static void launch_swiglu_out(OutQuant out, dim3 grids, dim3 blocks,
                              const KernelArgs& a) {
  switch (out) {
    case OUT_NONE:
      swiglu_oai_kernel<STORE, SCALE, OUT_NONE, BIAS><<<grids, blocks>>>(a);
      break;
    case OUT_FP8:
      swiglu_oai_kernel<STORE, SCALE, OUT_FP8, BIAS><<<grids, blocks>>>(a);
      break;
    case OUT_MXFP4:
      swiglu_oai_kernel<STORE, SCALE, OUT_MXFP4, BIAS><<<grids, blocks>>>(a);
      break;
    default: break;
  }
}

template <int BIAS>
static void launch_swiglu_bias(InDtype in, OutQuant out,
                               dim3 grids, dim3 blocks, const KernelArgs& a) {
  switch (in) {
    case IN_FP32:
      launch_swiglu_out<XS_F32, XSC_NONE, BIAS>(out, grids, blocks, a); break;
    case IN_FP16:
      launch_swiglu_out<XS_F16, XSC_NONE, BIAS>(out, grids, blocks, a); break;
    case IN_BF16:
      launch_swiglu_out<XS_BF16, XSC_NONE, BIAS>(out, grids, blocks, a); break;
    case IN_FP8:
      launch_swiglu_out<XS_FP8, XSC_TENSOR, BIAS>(out, grids, blocks, a); break;
    case IN_MXFP8:
      launch_swiglu_out<XS_FP8, XSC_MX, BIAS>(out, grids, blocks, a); break;
    case IN_FP4:
      launch_swiglu_out<XS_FP4, XSC_TENSOR, BIAS>(out, grids, blocks, a); break;
    case IN_MXFP4:
      launch_swiglu_out<XS_FP4, XSC_MX, BIAS>(out, grids, blocks, a); break;
    default: break;
  }
}

static void launch_swiglu(InDtype in, OutQuant out, BiasStore bias,
                          dim3 grids, dim3 blocks, const KernelArgs& a) {
  switch (bias) {
    case BS_NONE:
      launch_swiglu_bias<BS_NONE>(in, out, grids, blocks, a); break;
    case BS_F16:
      launch_swiglu_bias<BS_F16>(in, out, grids, blocks, a); break;
    case BS_BF16:
      launch_swiglu_bias<BS_BF16>(in, out, grids, blocks, a); break;
    case BS_F32:
      launch_swiglu_bias<BS_F32>(in, out, grids, blocks, a); break;
    default: break;
  }
}

// ---------------------------------------------------------------------------
// Host packing / unpacking
// ---------------------------------------------------------------------------

// Bytes needed to store `n` elements of the given input format.
static int64_t in_storage_bytes(InDtype dt, int64_t n) {
  const int bits = in_elem_bits(dt);
  return (bits == 4) ? (n + 1) / 2 : n * (bits / 8);
}

// quantize_dequantize_input() hands back dequantized floats; each is exactly
// representable in the target format, so re-encoding here is lossless and the
// device sees bit-identical values to the host reference. `rowLen` is the
// logical row length (2*dim); mx scales are indexed [row][block] because the
// blocks are cut inside a row.
static void pack_input(HipFp8Format fp8_format, InDtype dt, const float* xq,
                       const uint8_t* xs, float ts, int64_t n,
                       int64_t rowLen, void* dst) {
  const XStore store = x_store_of(dt);
  const XScale scaling = x_scale_of(dt);
  const int64_t bpr = mx_blocks_per_row(rowLen);

  auto scale_at = [&](int64_t i) -> float {
    if (scaling == XSC_MX) {
      const int64_t row = i / rowLen;
      const int64_t col = i - row * rowLen;
      return e8m0_to_float(xs[row * bpr + col / MX_BLOCK]);
    }
    return ts;
  };

  if (store == XS_FP4) {
    uint8_t* p = static_cast<uint8_t*>(dst);
    for (int64_t i = 0; i < n; i += 2) {
      const float scale = scale_at(i);
      const float hi = (i + 1 < n) ? xq[i + 1] / scale : 0.f;
      p[i >> 1] = e2m1x2_encode(xq[i] / scale, hi);
    }
    return;
  }

  for (int64_t i = 0; i < n; ++i) {
    switch (store) {
      case XS_F32:
        static_cast<float*>(dst)[i] = xq[i];
        break;
      case XS_F16:
        static_cast<__half*>(dst)[i] = __float2half(xq[i]);
        break;
      case XS_BF16:
        static_cast<__hip_bfloat16*>(dst)[i] = __float2bfloat16(xq[i]);
        break;
      default:
        static_cast<uint8_t*>(dst)[i] =
            hip_fp8_encode(xq[i] / scale_at(i), fp8_format);
        break;
    }
  }
}

static int bias_elem_bytes(BiasStore bias) {
  if (bias == BS_F16 || bias == BS_BF16) return 2;
  if (bias == BS_F32) return 4;
  return 0;
}

static void pack_bias(BiasStore bias, float* values, size_t n, void* dst) {
  if (bias == BS_NONE) {
    memset(values, 0, n * sizeof(float));
    return;
  }
  if (bias == BS_F16) {
    __half* packed = static_cast<__half*>(dst);
    for (size_t i = 0; i < n; ++i) {
      packed[i] = __float2half(values[i]);
      values[i] = __half2float(packed[i]);
    }
    return;
  }
  if (bias == BS_BF16) {
    __hip_bfloat16* packed = static_cast<__hip_bfloat16*>(dst);
    for (size_t i = 0; i < n; ++i) {
      packed[i] = __float2bfloat16(values[i]);
      values[i] = __bfloat162float(packed[i]);
    }
    return;
  }
  memcpy(dst, values, n * sizeof(float));
}

static int64_t out_storage_bytes(OutQuant q, int rows, int dim) {
  switch (q) {
    case OUT_NONE:
      return static_cast<int64_t>(rows) * dim * sizeof(float);
    case OUT_FP8:
      return static_cast<int64_t>(rows) * dim;
    case OUT_MXFP4:
      return static_cast<int64_t>(rows) * mxfp4_row_bytes(dim);
    default: return 0;
  }
}

// Decode what the kernel stored back into floats so it can be compared with
// the (also dequantized) host reference elementwise.
static void unpack_output(HipFp8Format fp8_format, OutQuant q,
                          int rows, int dim, const void* src,
                          const uint8_t* scales, float ts, float* dst) {
  if (q == OUT_NONE) {
    memcpy(dst, src, static_cast<size_t>(rows) * dim * sizeof(float));
    return;
  }
  if (q == OUT_FP8) {
    const uint8_t* p = static_cast<const uint8_t*>(src);
    for (int64_t i = 0; i < static_cast<int64_t>(rows) * dim; i++)
      dst[i] = hip_fp8_decode(p[i], fp8_format) * ts;
    return;
  }
  // OUT_MXFP4
  const int64_t blocksPerRow = mx_blocks_per_row(dim);
  const int64_t rowStride = mxfp4_row_bytes(dim);
  const uint8_t* p = static_cast<const uint8_t*>(src);
  for (int64_t r = 0; r < rows; r++) {
    for (int64_t d = 0; d < dim; d++) {
      const uint8_t nibble = static_cast<uint8_t>(
          (p[r * rowStride + (d >> 1)] >> ((d & 1) << 2)) & 0xf);
      dst[r * dim + d] =
          e2m1_decode(nibble) *
          e8m0_to_float(scales[r * blocksPerRow + (d / MX_BLOCK)]);
    }
  }
}

static void usage(const char* prog) {
  printf("Usage: %s <number of rows> <last dimension> <repeat> "
         "[in_dtype] [out_quant] [bias_dtype]\n", prog);
  printf("  in_dtype : fp32 | fp16 | bf16 | fp8 | mxfp8 | fp4 | mxfp4  (default fp32)\n");
  printf("  out_quant: none | fp8 | mxfp4                              (default none)\n");
  printf("  bias_dtype: none | fp16 | bf16 | fp32 "
         "(default none; fp32 is accuracy-only)\n");
}

int main(int argc, char* argv[])
{
  if (argc < 4 || argc > 7) {
    usage(argv[0]);
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
  if (!self_test_hip_rounding()) return 1;

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

  int device;
  hipDeviceProp_t device_properties;
  HIP_CHECK(hipGetDevice(&device));
  HIP_CHECK(hipGetDeviceProperties(&device_properties, device));
  const HipFp8Format fp8_format =
      hip_arch_uses_fnuz(device_properties.gcnArchName)
          ? HIP_FP8_FNUZ : HIP_FP8_OCP;

  printf("Shape of input tensor: ( %d %d )  in=%s out=%s bias=%s fp8=%s\n",
         rows, 2 * dim, in_dtype_name(in_dtype), out_quant_name(out_quant),
         bias_store_name(bias), hip_fp8_format_name(fp8_format));

  const int64_t x_nelems = static_cast<int64_t>(rows) * 2 * dim;
  const int64_t y_nelems = static_cast<int64_t>(rows) * dim;
  const int64_t b_nelems = static_cast<int64_t>(2) * dim;

  const int64_t b_bytes = b_nelems * sizeof(float);
  const int64_t b_store_bytes = b_nelems * bias_elem_bytes(bias);
  const int64_t x_store_bytes = in_storage_bytes(in_dtype, x_nelems);
  const int64_t y_store_bytes = out_storage_bytes(out_quant, rows, dim);

  // MX scales are [row][block] on both axes; blocks are cut inside a row.
  const int64_t x_blocksPerRow =
      mx_blocks_per_row(2 * static_cast<int64_t>(dim));
  const int64_t x_nscales = in_is_mx(in_dtype)
      ? static_cast<int64_t>(rows) * x_blocksPerRow : 0;
  const int64_t y_blocksPerRow = mx_blocks_per_row(dim);
  const int64_t y_nscales = out_is_mx(out_quant)
      ? static_cast<int64_t>(rows) * y_blocksPerRow : 0;

  float *X = static_cast<float*>(malloc(x_nelems * sizeof(float)));
  float *Xq = static_cast<float*>(malloc(x_nelems * sizeof(float)));
  float *B = static_cast<float*>(malloc(b_bytes));
  float *Y = static_cast<float*>(malloc(y_nelems * sizeof(float)));
  float *Y_ref = static_cast<float*>(malloc(y_nelems * sizeof(float)));
  // The reference activation before the epilogue quantization. values_match()
  // needs it: once quantized, every reference value sits exactly on a
  // representable code and so is always half a quantum from a boundary, which
  // says nothing about whether a rounding flip was plausible.
  float *Y_pre = static_cast<float*>(malloc(y_nelems * sizeof(float)));
  void *Xpacked = malloc(x_store_bytes);
  void *Bpacked = (bias == BS_NONE) ? nullptr : malloc(b_store_bytes);
  void *Ypacked = malloc(y_store_bytes);
  uint8_t *Xscales =
      static_cast<uint8_t*>(malloc(x_nscales ? x_nscales : 1));
  uint8_t *Yscales =
      static_cast<uint8_t*>(malloc(y_nscales ? y_nscales : 1));
  uint8_t *Yscales_ref =
      static_cast<uint8_t*>(malloc(y_nscales ? y_nscales : 1));

  if (!X || !Xq || !B || !Y || !Y_ref || !Y_pre || !Xpacked ||
      (bias != BS_NONE && !Bpacked) || !Ypacked || !Xscales ||
      !Yscales || !Yscales_ref) {
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
  pack_bias(bias, B, static_cast<size_t>(b_nelems), Bpacked);

  // Ground truth for the input axis: Xq holds the exact values the kernel must
  // reconstruct after dequantization.
  float xTensorScale = 1.f;
  quantize_dequantize_input_hip(
      fp8_format, in_dtype, X, x_nelems, 2 * static_cast<int64_t>(dim),
      Xq, Xscales, &xTensorScale);
  pack_input(fp8_format, in_dtype, Xq, Xscales, xTensorScale,
             x_nelems, 2 * static_cast<int64_t>(dim), Xpacked);

  void *d_X, *d_Y, *d_B = nullptr;
  uint8_t *d_Xs, *d_Ys;
  HIP_CHECK(hipMalloc(&d_X, x_store_bytes));
  if (bias != BS_NONE) {
    HIP_CHECK(hipMalloc(&d_B, b_store_bytes));
    HIP_CHECK(hipMemcpy(d_B, Bpacked, b_store_bytes, hipMemcpyHostToDevice));
  }
  HIP_CHECK(hipMalloc(&d_Y, y_store_bytes));
  HIP_CHECK(hipMalloc(&d_Xs, x_nscales ? x_nscales : 1));
  HIP_CHECK(hipMalloc(&d_Ys, y_nscales ? y_nscales : 1));

  HIP_CHECK(hipMemcpy(d_X, Xpacked, x_store_bytes, hipMemcpyHostToDevice));
  if (x_nscales)
    HIP_CHECK(hipMemcpy(d_Xs, Xscales, x_nscales, hipMemcpyHostToDevice));

  // Host reference: activation, then the epilogue quantization.
  ComputeSwigluOAI(rows, dim, kSwigluAlpha, kSwigluLimit, Xq, B, Y_ref);

  // A per-tensor fp8 output scale depends on the amax of the whole activation,
  // which a single-pass kernel cannot know; as in aiter it is supplied by the
  // caller. It is derived here exactly as quantize_dequantize_output() derives
  // it internally, so both sides quantize with the same scale.
  float yTensorScale = 1.f;
  if (out_quant == OUT_FP8) {
    float amax = 0.f;
    for (int64_t i = 0; i < y_nelems; i++) {
      const float v = fabsf(Y_ref[i]);
      if (v > amax) amax = v;
    }
    const float fp8_max =
        fp8_format == HIP_FP8_FNUZ ? kE4M3FnuzMax : kE4M3Max;
    yTensorScale = (amax > 0.f) ? (amax / fp8_max) : 1.f;
  }
  memcpy(Y_pre, Y_ref, static_cast<size_t>(y_nelems) * sizeof(float));
  quantize_dequantize_output_hip(
      fp8_format, out_quant, rows, dim, Y_ref, Yscales_ref);

  const int block_size = 256;
  const int64_t tilesPerRow = (dim + block_size - 1) / block_size;
  const int64_t totalTiles = static_cast<int64_t>(rows) * tilesPerRow;
  // The grid-stride loop lets the launch geometry stay inside the 32-bit grid
  // limit while the tile index, the element index and the guard remain int64.
  const int64_t maxBlocks = 1 << 20;
  dim3 grids(static_cast<unsigned int>(
      totalTiles < maxBlocks ? totalTiles : maxBlocks));
  dim3 blocks (block_size);

  KernelArgs args;
  args.rows = rows;
  args.dim = dim;
  args.alpha = kSwigluAlpha;
  args.limit = kSwigluLimit;
  args.X = d_X;
  args.Xs = d_Xs;
  args.xScale = xTensorScale;
  args.B = d_B;
  args.Y = d_Y;
  args.Ys = d_Ys;
  args.yScale = yTensorScale;
  args.xBlocksPerRow = x_blocksPerRow;
  args.yRowStrideBytes = mxfp4_row_bytes(dim);
  args.yBlocksPerRow = y_blocksPerRow;

  // check correctness before benchmarking
  launch_swiglu(in_dtype, out_quant, bias, grids, blocks, args);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipMemcpy(Ypacked, d_Y, y_store_bytes, hipMemcpyDeviceToHost));
  if (y_nscales)
    HIP_CHECK(hipMemcpy(Yscales, d_Ys, y_nscales, hipMemcpyDeviceToHost));
  unpack_output(fp8_format, out_quant, rows, dim, Ypacked, Yscales,
                yTensorScale, Y);

  // The tolerance is the baseline for the unquantized path; values_match() adds
  // one quantum of slack for a quantized element -- but only when the
  // pre-quantization reference sat near a rounding boundary -- using that
  // element's e8m0 block scale (mxfp4) or the per-tensor scale (fp8).
  const float tol = 1e-3f;
  bool ok = true;
  float maxdiff = 0.f;
  int64_t worst = 0;
  for (int64_t r = 0; r < rows; r++) {
    for (int64_t d = 0; d < dim; d++) {
      const int64_t i = r * dim + d;
      float scale = 1.f;
      if (out_quant == OUT_FP8)
        scale = yTensorScale;
      else if (out_quant == OUT_MXFP4)
        scale = e8m0_to_float(Yscales_ref[r * y_blocksPerRow + d / MX_BLOCK]);

      const float diff = fabsf(Y[i] - Y_ref[i]);
      if (diff > maxdiff) { maxdiff = diff; worst = i; }
      if (!hip_values_match(fp8_format, Y[i], Y_ref[i], Y_pre[i],
                            out_quant, scale, tol))
        ok = false;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) {
    int64_t scale_mismatch = 0;
    for (int64_t i = 0; i < y_nscales; i++)
      if (Yscales[i] != Yscales_ref[i]) scale_mismatch++;
    fprintf(stderr, "  max |device - reference| = %g at %ld (device %g, reference %g, tol %g)\n",
            maxdiff, static_cast<long>(worst), Y[worst], Y_ref[worst], tol);
    if (y_nscales)
      fprintf(stderr, "  mismatching e8m0 output scales: %ld of %ld\n",
              static_cast<long>(scale_mismatch),
              static_cast<long>(y_nscales));
  }

  if (bias == BS_F32) {
    printf("FP32 bias is accuracy-only; performance timing skipped.\n");
  } else {
    HIP_CHECK(hipDeviceSynchronize());
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      launch_swiglu(in_dtype, out_quant, bias, grids, blocks, args);
    }

    HIP_CHECK(hipDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    end - start).count();
    printf("Average execution time of SwiGLU (OAI) kernel: %f (us)\n",
           (time * 1e-3f) / repeat);
  }

  HIP_CHECK(hipFree(d_X));
  if (d_B != nullptr) HIP_CHECK(hipFree(d_B));
  HIP_CHECK(hipFree(d_Y));
  HIP_CHECK(hipFree(d_Xs));
  HIP_CHECK(hipFree(d_Ys));

  free(X);
  free(Xq);
  free(B);
  free(Y);
  free(Y_ref);
  free(Y_pre);
  free(Xpacked);
  free(Bpacked);
  free(Ypacked);
  free(Xscales);
  free(Yscales);
  free(Yscales_ref);

  return EXIT_SUCCESS;
}
