#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <chrono>
#include <random>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#if !defined(__CUDACC_VER_MAJOR__) || !defined(__CUDACC_VER_MINOR__)
#error "swiglu-oai-cuda must be compiled with the NVIDIA CUDA compiler"
#elif !defined(CUDART_VERSION) || (CUDART_VERSION < 11080)
#error "swiglu-oai-cuda requires CUDA Toolkit 11.8 or newer for FP8 support"
#endif

// cuda_fp4.h and __nv_cvt_float_to_e8m0() first ship with CUDA Toolkit 12.8.
// Older toolkits keep every other format and take the software E2M1/E8M0
// codecs below, which make the same rounding decisions as reference.h.
#ifndef SWIGLU_HAS_NATIVE_FP4
#if CUDART_VERSION >= 12080
#define SWIGLU_HAS_NATIVE_FP4 1
#else
#define SWIGLU_HAS_NATIVE_FP4 0
#endif
#endif

#if SWIGLU_HAS_NATIVE_FP4
#include <cuda_fp4.h>
#endif
#include "reference.h"

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    const cudaError_t error_ = (call);                                         \
    if (error_ != cudaSuccess) {                                               \
      fprintf(stderr, "CUDA error at %s:%d: %s failed: %s\n",                  \
              __FILE__, __LINE__, #call, cudaGetErrorString(error_));          \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

static constexpr int WarpSize = 32;
static_assert(MX_BLOCK == WarpSize,
              "The MXFP4 kernel requires one 32-lane warp per MX block");
static_assert(sizeof(size_t) >= 8,
              "This benchmark requires 64-bit size_t indexing");

// ---------------------------------------------------------------------------
// Native low-precision helpers, shared by host and device
// ---------------------------------------------------------------------------

#if !SWIGLU_HAS_NATIVE_FP4
// Software E2M1 codec for toolkits without cuda_fp4.h. Representable
// magnitudes are 0, 0.5, 1, 1.5, 2, 3, 4, 6; the nearest-value search with its
// tie-to-even-index rule mirrors round_to_e2m1() in reference.h, including the
// strict `> 5.f` that lets the 4/6 tie land on 4.
__host__ __device__ __forceinline__
uint8_t soft_e2m1_encode(const float x)
{
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
}

__host__ __device__ __forceinline__
float soft_e2m1_decode(const uint8_t nibble)
{
  const float lut[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
  const float m = lut[nibble & 0x7u];
  return (nibble & 0x8u) ? -m : m;
}
#endif

// Two E2M1 values packed into one byte, low nibble first. Raw byte storage
// keeps the host packing independent of the toolkit's FP4 type.
__host__ __device__ __forceinline__
uint8_t fp4x2_pack(const float lo, const float hi)
{
#if SWIGLU_HAS_NATIVE_FP4
  return __nv_cvt_float2_to_fp4x2(
      make_float2(lo, hi), __NV_E2M1, cudaRoundNearest);
#else
  return static_cast<uint8_t>(soft_e2m1_encode(lo) |
                              (soft_e2m1_encode(hi) << 4));
#endif
}

__host__ __device__ __forceinline__
float fp4x2_unpack(const uint8_t byte, const int hi)
{
#if SWIGLU_HAS_NATIVE_FP4
  const __half2_raw h2 = __nv_cvt_fp4x2_to_halfraw2(byte, __NV_E2M1);
  __half_raw h;
  h.x = hi ? h2.y : h2.x;
  return __half2float(__half(h));
#else
  return soft_e2m1_decode(hi ? (byte >> 4) : (byte & 0xfu));
#endif
}

// Round-toward-zero E8M0 encode of amax / 2^kE2M1MaxExp, which is the MXFP4
// block scale. For a positive normal float the biased IEEE-754 exponent byte
// minus kE2M1MaxExp is exactly the wanted byte; subnormals clamp to 0.
__device__ __forceinline__
uint8_t mxfp4_scale_byte(const float amax)
{
  if (!(amax > 0.f)) return 127;  // E8M0 has no zero; 2^0 for an empty block
#if SWIGLU_HAS_NATIVE_FP4
  return __nv_cvt_float_to_e8m0(amax * 0.25f, __NV_SATFINITE, cudaRoundZero);
#else
  const int s = static_cast<int>((__float_as_uint(amax) >> 23) & 0xffu) -
                kE2M1MaxExp;
  return s < 0 ? 0 : s > 254 ? 254 : static_cast<uint8_t>(s);
#endif
}

// FP32 reserves exponent 0 for zero/subnormals while E8M0 treats exponent 0 as 2^{-127}
// TODO: b = 0xFF
__host__ __device__ __forceinline__
float e8m0_decode(const uint8_t b)
{
  const uint32_t bits = b ? static_cast<uint32_t>(b) << 23 : 0x00400000u;
#ifdef __CUDA_ARCH__
  return __uint_as_float(bits);
#else
  float value;
  memcpy(&value, &bits, sizeof(value));
  return value;
#endif
}

// expf(-alpha*gate) overflows for large negative gate.
__device__ __forceinline__ float dev_sigmoid(const float x)
{
  if (x >= 0.f) {
    return 1.f / (1.f + expf(-x));
  } else {
    const float ex = expf(x);
    return ex / (1.f + ex);
  }
}

// Input load: fetch the native element and undo the quantization scale.

template <int STORE>
__device__ __forceinline__ float load_x_raw(const void* __restrict__ X,
                                            const size_t i)
{
  if constexpr (STORE == XS_F32) {
    return static_cast<const float*>(X)[i];
  } else if constexpr (STORE == XS_F16) {
    return __half2float(static_cast<const __half*>(X)[i]);
  } else if constexpr (STORE == XS_BF16) {
    return __bfloat162float(static_cast<const __nv_bfloat16*>(X)[i]);
  } else if constexpr (STORE == XS_FP8) {
    return static_cast<float>(static_cast<const __nv_fp8_e4m3*>(X)[i]);
  } else {
    return fp4x2_unpack(static_cast<const uint8_t*>(X)[i >> 1],
                        static_cast<int>(i & 1));
  }
}

// Load one adjacent gate/linear pair. FP4 stores the pair in one byte and FP8
// stores it in one aligned 16-bit word, so one vector conversion decodes both
// values. Other formats retain their ordinary scalar loads.
template <int STORE, int SCALE>
__device__ __forceinline__ float2 load_x_pair(
    const void* __restrict__ X,
    const uint8_t* __restrict__ xscales,
    const float xtscale,
    const size_t i,
    const size_t scaleIdx)
{
  float2 pair;
  if constexpr (STORE == XS_FP4) {
    const uint8_t packed = static_cast<const uint8_t*>(X)[i >> 1];
#if SWIGLU_HAS_NATIVE_FP4
    const __half2_raw h2 = __nv_cvt_fp4x2_to_halfraw2(packed, __NV_E2M1);
    __half_raw lo, hi;
    lo.x = h2.x;
    hi.x = h2.y;
    pair = make_float2(__half2float(__half(lo)), __half2float(__half(hi)));
#else
    pair = make_float2(fp4x2_unpack(packed, 0), fp4x2_unpack(packed, 1));
#endif
  } else if constexpr (STORE == XS_FP8) {
    const __nv_fp8x2_storage_t packed =
        static_cast<const __nv_fp8x2_e4m3*>(X)[i >> 1].__x;
    const __half2_raw h2 = __nv_cvt_fp8x2_to_halfraw2(packed, __NV_E4M3);
    __half_raw lo, hi;
    lo.x = h2.x;
    hi.x = h2.y;
    pair = make_float2(__half2float(__half(lo)), __half2float(__half(hi)));
  } else {
    pair = make_float2(load_x_raw<STORE>(X, i),
                       load_x_raw<STORE>(X, i + 1));
  }

  float scale = 1.f;
  if constexpr (SCALE == XSC_TENSOR) scale = xtscale;
  else if constexpr (SCALE == XSC_MX)
    scale = e8m0_decode(xscales[scaleIdx]);
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
                                           const int d)
{
  if constexpr (BIAS == BS_NONE) {
    (void)B;
    (void)d;
    return 0.f;
  } else if constexpr (BIAS == BS_F16) {
    return __half2float(static_cast<const __half*>(B)[d]);
  } else if constexpr (BIAS == BS_BF16) {
    return __bfloat162float(static_cast<const __nv_bfloat16*>(B)[d]);
  } else {
    return static_cast<const float*>(B)[d];
  }
}

// Compute one output value from one adjacent gate/linear pair.
//
// Each input row contains 2*dim values in this order:
//
//   gate[0], linear[0], gate[1], linear[1], ...
//
// Therefore output column d reads input columns 2*d and 2*d+1. Low-precision
// input values are first converted to float and multiplied by either their
// tensor-wide scale or their 32-value MX-block scale. Bias is then added in
// float. The gate is capped only above, while the linear value is capped on
// both sides:
//
//   y = gate * sigmoid(alpha * gate) * (linear + 1)
//
// This helper is shared by every output path.
template <int STORE, int SCALE, int BIAS>
__device__ __forceinline__ float swiglu_elem(
    const KernelArgs& a, const int64_t row, const int64_t d)
{
  const int64_t index = row * a.dim + d;
  const int64_t xIndex = 2 * index;
  // 2*d is even and 2*d+1 is odd, so the pair can never be split across an
  // MX_BLOCK boundary: one scale lookup covers both halves.
  int64_t scaleIdx = 0;
  if constexpr (SCALE == XSC_MX) {
    // LLM dimensions are normally multiples of 16. In that case input MX
    // blocks have no row padding, so the output's flat index maps directly to
    // its input scale. Retain the row-aware path for arbitrary dimensions.
    scaleIdx = (a.dim % (MX_BLOCK / 2) == 0)
             ? index / (MX_BLOCK / 2)
             : row * a.xBlocksPerRow + (2 * d) / MX_BLOCK;
  }

  const float2 pair = load_x_pair<STORE, SCALE>(
      a.X, a.Xs, a.xScale, xIndex, scaleIdx);
  float gate = pair.x + load_bias<BIAS>(a.B, 2 * d);
  float linear = pair.y + load_bias<BIAS>(a.B, 2 * d + 1);

  gate = fminf(gate, a.limit);
  linear = fminf(fmaxf(linear, -a.limit), a.limit);
  return gate * dev_sigmoid(a.alpha * gate) * (linear + 1.f);
}

/*
  Description of the fused kernel:

  Each input row contains interleaved gate and linear values:

  gate[0], linear[0], gate[1], linear[1], ...

  For output column d, it computes:

  gate   = min(input[2*d] + bias[2*d], limit)
  linear = clamp(input[2*d+1] + bias[2*d+1], -limit, limit)
  output[d] = gate × sigmoid(alpha × gate) × (linear + 1)

  Thus an input shaped [rows, 2 × dim] produces [rows, dim].

  Input loading

  The kernel supports FP32, FP16, BF16, FP8, and FP4 storage.

  1. It loads the stored value.
  2. FP8/FP4 values are converted to float.
  3.1 For tensor-scaled formats, it multiplies by one tensor-wide scale.
  3.2 For MX formats, it multiplies by the E8M0 scale belonging to that 32-value block.
  4. All activation arithmetic then happens in FP32.

  FP4 stores two values per byte, so the unpacking helper selects either the low or high value.

  Bias handling

  Bias is selected at compile time:

  • none: the compiler completely removes the bias load.
  • fp16: loaded and converted to float.
  • bf16: loaded and converted to float.
  • fp32: loaded directly, but used only for accuracy testing.

  This avoids a runtime branch for every output element.

  FP32/FP8 output path

  A single fused kernel serves every output path (matching the HIP and SYCL
  ports); one CUDA thread computes one output element.

  The thread maps its tiled index to a row and column, loads the gate/linear pair, evaluates SwiGLU, and
  then:

  • writes the float directly for FP32 output, or
  • divides by the tensor-wide output scale, rounds to E4M3, and writes FP8.

  Threads are tiled as (row, block of blockDim.x columns) rather than laid out flat, so that an MXFP4 group of
  32 values always lands on one 32-lane warp. A grid-stride loop over int64 tile ids lets each warp process
  additional groups when the tensor exceeds the launched grid, and keeps the index and guard 64-bit safe.

  MXFP4 output path

  MXFP4 requires one shared scale for every group of 32 output values. Therefore, one 32-thread CUDA warp owns
  one MXFP4 group.

  1. Lane 0 computes value 0, lane 1 computes value 1, and so forth.
  2. Missing values in a partial final group are represented as zero.
  3. The warp finds the largest absolute value using five shuffle-reduction steps: offsets 16, 8, 4, 2, and
     1.
  4. Lane 0 converts that maximum’s exponent into an E8M0 scale byte and stores it.
  5. Lane 0 broadcasts the scale through warp registers.
  6. Every value is divided by the shared scale and rounded to E2M1.
  7. Even-numbered lanes pack adjacent pairs:
    • lane 0 packs values 0 and 1
    • lane 2 packs values 2 and 3
    • …
    • lane 30 packs values 30 and 31

  The 16 even lanes therefore write 16 bytes containing 32 FP4 values.
*/

template <int STORE, int SCALE, OutQuant OUT, int BIAS>
__global__ void swiglu_oai_kernel(const KernelArgs a)
{
  const int64_t tilesPerRow =
      (static_cast<int64_t>(a.dim) + blockDim.x - 1) / blockDim.x;
  const int64_t totalTiles = static_cast<int64_t>(a.rows) * tilesPerRow;
  const unsigned int mask = 0xffffffffu;

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
        static_cast<__nv_fp8_e4m3*>(a.Y)[
            row * static_cast<int64_t>(a.dim) + d].__x =
            __nv_cvt_float_to_fp8(y / a.yScale, __NV_SATFINITE, __NV_E4M3);
    } else {  // OUT_MXFP4
      // amax over the 32 lanes of this mx block; inactive lanes contribute 0.
      float amax = active ? fabsf(y) : 0.f;
      #pragma unroll
      for (int m = 1; m < WarpSize; m <<= 1)
        amax = fmaxf(amax, __shfl_xor_sync(mask, amax, m, WarpSize));

      const int lane = threadIdx.x & (WarpSize - 1);
      // Dividing by E2M1's largest power-of-two exponent before conversion
      // chooses the block scale. E8M0 round-toward-zero is floor(log2) for
      // positive values, exactly matching reference.h's scale rule. E8M0 has
      // no zero, so retain byte 127 (scale 1) for an all-zero/NaN block.
      const uint8_t sb = mxfp4_scale_byte(amax);
      const float s = e8m0_decode(sb);

      // Lane 0 of the block owns the scale byte. Its column is in range
      // whenever the block exists, so an inactive lane 0 means nothing to do.
      if (lane == 0 && active)
        a.Ys[row * a.yBlocksPerRow + (d / MX_BLOCK)] = sb;

      // Two fp4 values share a byte, so the even lane of each pair packs its
      // own value with its right-hand neighbour's (width 2 keeps adjacent
      // pairs independent) and performs the single store.
      const float paired = __shfl_down_sync(mask, y, 1, 2);
      if (active && (d & 1) == 0) {
        const float hi =
            (d + 1 < static_cast<int64_t>(a.dim)) ? paired : 0.f;
        static_cast<uint8_t*>(a.Y)[row * a.yRowStrideBytes + (d >> 1)] =
            fp4x2_pack(y / s, hi / s);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Launch dispatch. The switch runs on the host and costs nothing next to a
// kernel launch, so the timing loop can call it directly.
// ---------------------------------------------------------------------------

template <int STORE, int SCALE, int BIAS>
static void launch_swiglu_out(OutQuant out, dim3 grids, dim3 blocks,
                              const KernelArgs& a) {
  switch (out) {
    case OUT_NONE:
      swiglu_oai_kernel<STORE, SCALE, OUT_NONE, BIAS>
          <<<grids, blocks>>>(a);
      break;
    case OUT_FP8:
      swiglu_oai_kernel<STORE, SCALE, OUT_FP8, BIAS>
          <<<grids, blocks>>>(a);
      break;
    case OUT_MXFP4:
      swiglu_oai_kernel<STORE, SCALE, OUT_MXFP4, BIAS>
          <<<grids, blocks>>>(a);
      break;
    default:
      break;
  }
}

template <int BIAS>
static void launch_swiglu_bias(InDtype in, OutQuant oq,
                               dim3 grids, dim3 blocks, const KernelArgs& a)
{
  switch (in) {
    case IN_FP32:
      launch_swiglu_out<XS_F32, XSC_NONE, BIAS>(oq, grids, blocks, a); break;
    case IN_FP16:
      launch_swiglu_out<XS_F16, XSC_NONE, BIAS>(oq, grids, blocks, a); break;
    case IN_BF16:
      launch_swiglu_out<XS_BF16, XSC_NONE, BIAS>(oq, grids, blocks, a); break;
    case IN_FP8:
      launch_swiglu_out<XS_FP8, XSC_TENSOR, BIAS>(oq, grids, blocks, a); break;
    case IN_MXFP8:
      launch_swiglu_out<XS_FP8, XSC_MX, BIAS>(oq, grids, blocks, a); break;
    case IN_FP4:
      launch_swiglu_out<XS_FP4, XSC_TENSOR, BIAS>(oq, grids, blocks, a); break;
    case IN_MXFP4:
      launch_swiglu_out<XS_FP4, XSC_MX, BIAS>(oq, grids, blocks, a); break;
    default: break;
  }
}

static void launch_swiglu(InDtype in, OutQuant oq, BiasStore bias,
                          dim3 grids, dim3 blocks, const KernelArgs& a)
{
  switch (bias) {
    case BS_NONE: launch_swiglu_bias<BS_NONE>(in, oq, grids, blocks, a); break;
    case BS_F16:  launch_swiglu_bias<BS_F16>(in, oq, grids, blocks, a);  break;
    case BS_BF16: launch_swiglu_bias<BS_BF16>(in, oq, grids, blocks, a); break;
    case BS_F32:  launch_swiglu_bias<BS_F32>(in, oq, grids, blocks, a);  break;
    default: break;
  }
}

// ---------------------------------------------------------------------------
// Host packing: turn the dequantized floats reference.h handed back into the
// native storage the kernel reads. Dividing by the same scale reference.h
// multiplied by lands on an exactly representable value, so the native
// conversion here is lossless and host and device see identical numbers.
// ---------------------------------------------------------------------------

static void pack_input(InDtype dtype, const float* xq, size_t n,
                       size_t rowLen, const uint8_t* scales, float tscale,
                       void* dst)
{
  const XStore store = x_store_of(dtype);
  const XScale sc = x_scale_of(dtype);
  const size_t bpr = mx_blocks_per_row(rowLen);

  // Scale for element `i`, using the same [row][block] layout reference.h
  // filled in.
  auto scale_at = [&](size_t i) -> float {
    if (sc == XSC_MX) {
      const size_t r = i / rowLen;
      const size_t c = i - r * rowLen;
      return e8m0_decode(scales[r * bpr + c / MX_BLOCK]);
    }
    return tscale;
  };

  if (store == XS_FP4) {
    // rowLen is 2*dim and therefore even, so a packed pair never straddles a
    // row, and 2j/2j+1 never straddle an MX block either: one scale for both.
    uint8_t* p = static_cast<uint8_t*>(dst);
    for (size_t j = 0; j < n / 2; j++) {
      const size_t i = 2 * j;
      const float s = scale_at(i);
      p[j] = fp4x2_pack(xq[i] / s, xq[i + 1] / s);
    }
    return;
  }

  for (size_t i = 0; i < n; i++) {
    switch (store) {
      case XS_F32:
        static_cast<float*>(dst)[i] = xq[i];
        break;
      case XS_F16:
        static_cast<__half*>(dst)[i] = __float2half(xq[i]);
        break;
      case XS_BF16:
        static_cast<__nv_bfloat16*>(dst)[i] = __float2bfloat16(xq[i]);
        break;
      default:  // XS_FP8
        static_cast<__nv_fp8_e4m3*>(dst)[i] = __nv_fp8_e4m3(xq[i] / scale_at(i));
        break;
    }
  }
}

static int bias_elem_bytes(BiasStore bias)
{
  if (bias == BS_F16 || bias == BS_BF16) return 2;
  if (bias == BS_F32) return 4;
  return 0;
}

// Pack the bias in its production storage type and replace the FP32 reference
// values with the exact values that the device will decode.
static void pack_bias(BiasStore bias, float* values, size_t n, void* dst)
{
  if (bias == BS_NONE) {
    memset(values, 0, n * sizeof(float));
    return;
  }
  if (bias == BS_F16) {
    __half* packed = static_cast<__half*>(dst);
    for (size_t i = 0; i < n; i++) {
      packed[i] = __float2half(values[i]);
      values[i] = __half2float(packed[i]);
    }
    return;
  }
  if (bias == BS_BF16) {
    __nv_bfloat16* packed = static_cast<__nv_bfloat16*>(dst);
    for (size_t i = 0; i < n; i++) {
      packed[i] = __float2bfloat16(values[i]);
      values[i] = __bfloat162float(packed[i]);
    }
    return;
  }
  memcpy(dst, values, n * sizeof(float));
}

// Undo the device's output quantization so the result can be compared with
// reference.h's dequantized reference elementwise.
static void unpack_output(OutQuant q, int rows, int dim, size_t blocksPerRow,
                          const void* src, const uint8_t* yscales,
                          float youtscale, float* Y)
{
  const size_t n = static_cast<size_t>(rows) * dim;

  if (q == OUT_NONE) {
    memcpy(Y, src, n * sizeof(float));
    return;
  }

  if (q == OUT_FP8) {
    const __nv_fp8_e4m3* p = static_cast<const __nv_fp8_e4m3*>(src);
    for (size_t i = 0; i < n; i++) Y[i] = float(p[i]) * youtscale;
    return;
  }

  const uint8_t* p = static_cast<const uint8_t*>(src);
  const size_t rowBytes = mxfp4_row_bytes(dim);
  for (size_t i = 0; i < static_cast<size_t>(rows); i++) {
    for (size_t b = 0; b < blocksPerRow; b++) {
      const float s = e8m0_decode(yscales[i * blocksPerRow + b]);
      const int d0 = static_cast<int>(b * MX_BLOCK);
      const int nk = (d0 + MX_BLOCK < dim) ? MX_BLOCK : (dim - d0);
      const uint8_t* blk = p + i * rowBytes + b * (MX_BLOCK / 2);
      for (int k = 0; k < nk; k++) {
        Y[i * dim + d0 + k] = fp4x2_unpack(blk[k / 2], k & 1) * s;
      }
    }
  }
}

static void usage(const char* prog)
{
  printf("Usage: %s <number of rows> <last dimension> <repeat> "
         "[in_dtype] [out_quant] [bias_dtype]\n", prog);
  printf("  in_dtype  : fp32 fp16 bf16 fp8 mxfp8 fp4 mxfp4 (default fp32)\n");
  printf("  out_quant : none fp8 mxfp4 (default none)\n");
  printf("  bias_dtype: none fp16 bf16 fp32 (default none; fp32 is accuracy-only)\n");
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
  if (!self_test_rounding()) return 1;

  const InDtype indt = (argc > 4) ? parse_in_dtype(argv[4]) : IN_FP32;
  if (indt == IN_INVALID) {
    fprintf(stderr, "Error: invalid input dtype '%s' "
            "(valid: fp32 fp16 bf16 fp8 mxfp8 fp4 mxfp4)\n", argv[4]);
    return 1;
  }

  const OutQuant oq = (argc > 5) ? parse_out_quant(argv[5]) : OUT_NONE;
  if (oq == OUT_INVALID) {
    fprintf(stderr, "Error: invalid output quantization '%s' "
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
         rows, 2 * dim, in_dtype_name(indt), out_quant_name(oq),
         bias_store_name(bias));

  const size_t y_nelems = static_cast<size_t>(rows) * dim;
  const size_t x_nelems = y_nelems * 2;
  const size_t b_nelems = static_cast<size_t>(2) * dim; // bias

  const size_t x_bytes = x_nelems * sizeof(float);
  const size_t y_bytes = y_nelems * sizeof(float);
  const size_t b_bytes = b_nelems * sizeof(float);
  const size_t b_store_bytes = b_nelems * bias_elem_bytes(bias);

  // Native input storage. Input MX scales are [row][block] over rows of
  // 2*dim elements, with blocks cut inside a row.
  const size_t x_rowlen = b_nelems;  // 2*dim
  const size_t x_bpr = mx_blocks_per_row(x_rowlen);
  const size_t x_store_bytes = (in_elem_bits(indt) == 4)
                             ? (x_nelems + 1) / 2
                             : x_nelems * (in_elem_bits(indt) / 8);
  const size_t x_nscales =
      in_is_mx(indt) ? static_cast<size_t>(rows) * x_bpr : 0;

  // mxfp4 output rows are padded to a whole number of MX blocks so no byte is
  // ever shared between two rows; mxfp4_row_bytes() is the layout every
  // backend agrees on.
  const size_t blocksPerRow = mx_blocks_per_row(dim);
  const size_t y_row_bytes = mxfp4_row_bytes(dim);
  const size_t y_nscales =
      out_is_mx(oq) ? static_cast<size_t>(rows) * blocksPerRow : 0;
  const size_t y_store_bytes = (oq == OUT_NONE) ? y_bytes
                            : (oq == OUT_FP8)  ? y_nelems
                            : static_cast<size_t>(rows) * y_row_bytes;

  float *X = static_cast<float*>(malloc(x_bytes));
  float *B = static_cast<float*>(malloc(b_bytes));
  float *Xq = static_cast<float*>(malloc(x_bytes));
  float *Y = static_cast<float*>(malloc(y_bytes));
  float *Y_ref = static_cast<float*>(malloc(y_bytes));
  float *Y_pre = static_cast<float*>(malloc(y_bytes));

  void *Xp = malloc(x_store_bytes);
  void *Bp = (bias == BS_NONE) ? nullptr : malloc(b_store_bytes);
  void *Yp = malloc(y_store_bytes);

  uint8_t *Xs = static_cast<uint8_t*>(malloc(x_nscales ? x_nscales : 1));
  uint8_t *Ys = static_cast<uint8_t*>(malloc(y_nscales ? y_nscales : 1));
  uint8_t *Ys_ref =
      static_cast<uint8_t*>(malloc(y_nscales ? y_nscales : 1));

  if (!X || !B || !Xq || !Y || !Y_ref || !Y_pre ||
      !Xp || (bias != BS_NONE && !Bp) || !Yp ||
      !Xs || !Ys || !Ys_ref) {
    fprintf(stderr, "Error: host allocation failed\n");
    return EXIT_FAILURE;
  }

  std::default_random_engine generator(123);
  // both clamped and unclamped elements are exercised.
  std::uniform_real_distribution<float> distribution(-12.f, 12.f);

  for (size_t i = 0; i < x_nelems; i++) {
    X[i] = distribution(generator);
  }
  for (size_t i = 0; i < b_nelems; i++) {
    B[i] = distribution(generator);
  }
  pack_bias(bias, B, b_nelems, Bp);

  // Quantize FP32 values and then pack them
  float xtscale = 1.f;
  quantize_dequantize_input(indt, X, x_nelems, x_rowlen, Xq, Xs, &xtscale);
  pack_input(indt, Xq, x_nelems, x_rowlen, Xs, xtscale, Xp);

  void *d_X, *d_Y, *d_B = nullptr;
  uint8_t *d_Xs, *d_Ys;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_X), x_store_bytes));
  CUDA_CHECK(cudaMemcpy(d_X, Xp, x_store_bytes, cudaMemcpyHostToDevice));
  if (bias != BS_NONE) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_B), b_store_bytes));
    CUDA_CHECK(cudaMemcpy(d_B, Bp, b_store_bytes, cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&d_Xs), x_nscales ? x_nscales : 1));
  if (x_nscales)
    CUDA_CHECK(cudaMemcpy(d_Xs, Xs, x_nscales, cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_Y), y_store_bytes));
  CUDA_CHECK(cudaMalloc(
      reinterpret_cast<void**>(&d_Ys), y_nscales ? y_nscales : 1));

  ComputeSwigluOAI(rows, dim, kSwigluAlpha, kSwigluLimit, Xq, B, Y_ref);

  // The fp8 fused quant uses a single per-tensor scale. A kernel cannot see
  // the whole tensor in its epilogue, so the scale is computed up front and
  // handed to the kernel, which is how aiter's per-tensor quant works too.
  // This is the same amax/448 reference.h derives internally.
  float youtscale = 1.f;
  if (oq == OUT_FP8) {
    float amax = 0.f;
    for (size_t i = 0; i < y_nelems; i++) {
      const float a = fabsf(Y_ref[i]);
      if (a > amax) amax = a;
    }
    youtscale = (amax > 0.f) ? (amax / kE4M3Max) : 1.f;
  }

  // values_match() needs the reference as it was BEFORE the output quant: once
  // quantized every value sits exactly on a representable code, which says
  // nothing about how close the activation came to a rounding boundary.
  memcpy(Y_pre, Y_ref, y_bytes);
  quantize_dequantize_output(oq, rows, dim, Y_ref, Ys_ref);

  const int block_size = 256;
  // Threads are tiled (row, block of block_size columns) so an MXFP4 group of
  // 32 values always lands on one warp. The grid-stride loop then covers any
  // excess, keeping grid.x inside its hardware limit for very large tensors.
  const int64_t tilesPerRow =
      (static_cast<int64_t>(dim) + block_size - 1) / block_size;
  const int64_t totalTiles = static_cast<int64_t>(rows) * tilesPerRow;
  const int64_t max_blocks = 1 << 20;
  size_t nblocks = static_cast<size_t>(
      totalTiles < max_blocks ? totalTiles : max_blocks);

  // Kernel launch arguments
  KernelArgs a;
  a.rows = rows;
  a.dim = dim;
  a.alpha = kSwigluAlpha;
  a.limit = kSwigluLimit;
  a.X = d_X;
  a.Xs = d_Xs;
  a.xScale = xtscale;
  a.B = d_B;
  a.Y = d_Y;
  a.Ys = d_Ys;
  a.yScale = youtscale;
  a.xBlocksPerRow = static_cast<int64_t>(x_bpr);
  a.yRowStrideBytes = static_cast<int64_t>(y_row_bytes);
  a.yBlocksPerRow = static_cast<int64_t>(blocksPerRow);
  const dim3 grids(static_cast<unsigned int>(nblocks));
  const dim3 blocks(block_size);

  // check correctness before benchmarking
  launch_swiglu(indt, oq, bias, grids, blocks, a);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(Yp, d_Y, y_store_bytes, cudaMemcpyDeviceToHost));

  if (y_nscales)
    CUDA_CHECK(cudaMemcpy(Ys, d_Ys, y_nscales, cudaMemcpyDeviceToHost));
  unpack_output(oq, rows, dim, blocksPerRow, Yp, Ys, youtscale, Y);

  const float tol = 1e-3f;
  bool ok = true;
  float maxdiff = 0.f;
  size_t worst = 0;
  for (size_t i = 0; i < y_nelems; i++) {
    float scale = youtscale;
    if (oq == OUT_MXFP4) {
      const size_t r = i / dim;
      const size_t b = (i - r * dim) / MX_BLOCK;
      scale = e8m0_to_float(Ys_ref[r * blocksPerRow + b]);
    }
    const float diff = fabsf(Y[i] - Y_ref[i]);
    if (diff > maxdiff) {
      maxdiff = diff;
      worst = i;
    }
    if (!values_match(Y[i], Y_ref[i], Y_pre[i], oq, scale, tol))
      ok = false;
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) {
    size_t scale_mismatch = 0;
    for (size_t i = 0; i < y_nscales; ++i)
      if (Ys[i] != Ys_ref[i]) ++scale_mismatch;
    fprintf(stderr, "  max |device - reference| = %g at %zu "
            "(device %g, reference %g, tol %g)\n",
            maxdiff, worst, Y[worst], Y_ref[worst], tol);
    if (y_nscales)
      fprintf(stderr, "  mismatching e8m0 output scales: %zu of %zu\n",
              scale_mismatch, y_nscales);
  }

  if (bias == BS_F32) {
    printf("FP32 bias is accuracy-only; performance timing skipped.\n");
  } else {
    CUDA_CHECK(cudaDeviceSynchronize());
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      launch_swiglu(indt, oq, bias, grids, blocks, a);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    const auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                          end - start).count();
    printf("Average execution time of SwiGLU (OAI) kernel: %f (us)\n",
           (time * 1e-3f) / repeat);
  }

  CUDA_CHECK(cudaFree(d_X));
  if (d_B != nullptr) CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_Y));
  CUDA_CHECK(cudaFree(d_Xs));
  CUDA_CHECK(cudaFree(d_Ys));

  free(X);
  free(B);
  free(Xq);
  free(Y);
  free(Y_ref);
  free(Y_pre);
  free(Xp);
  free(Bp);
  free(Yp);
  free(Xs);
  free(Ys);
  free(Ys_ref);

  return EXIT_SUCCESS;
}
