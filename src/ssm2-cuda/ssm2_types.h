// Mixed-precision types of the TileGym Mamba-2 forward pass
// (src/tilegym/ops/cutile/mamba2.py, mamba2_chunk_forward):
//
//   BF16  x, B, C, dt, out, and the chunk-entry states (state_dtype = c.dtype)
//   FP32  a, D, decay cumsum, init_state, final_state, and every accumulator
//
// GEMM operands are rounded where the cuTile kernels cast them:
//   state  X^T @ bf16(B * dt * exp2(d_last - d))   BF16 x BF16 -> FP32
//   C B^T  C @ B^T                                  BF16 x BF16 -> FP32
//   C H^T  C @ H^T                                  BF16 x BF16 -> FP32
//   out    tf32(masked CB) @ tf32(bf16(dt * x))     TF32 x TF32 -> FP32
//
// Each BF16 or TF32 product is exact in FP32, so scalar FP32 FMAs reproduce
// the tensor-core products; only the accumulation order differs.

#pragma once

#include <stdint.h>
#include <string.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define SSM2_HD __host__ __device__
#else
#define SSM2_HD
#endif

// A distinct type keeps BF16 bits from converting implicitly to arithmetic.
struct ssm2_bf16 {
  uint16_t bits;
};

struct alignas(8) ssm2_bf16x4 {
  ssm2_bf16 x, y, z, w;
};

SSM2_HD inline uint32_t ssm2_float_bits(float v) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return __float_as_uint(v);
#else
  uint32_t u;
  memcpy(&u, &v, sizeof(u));
  return u;
#endif
}

SSM2_HD inline float ssm2_bits_float(uint32_t u) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
  return __uint_as_float(u);
#else
  float v;
  memcpy(&v, &u, sizeof(v));
  return v;
#endif
}

// Round to nearest even, as torch and __float2bfloat16_rn do; NaN stays NaN.
SSM2_HD inline ssm2_bf16 ssm2_to_bf16(float v) {
  const uint32_t u = ssm2_float_bits(v);
  if ((u & 0x7fffffffu) > 0x7f800000u)
    return {(uint16_t)((u >> 16) | 0x40u)};
  return {(uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16)};
}

SSM2_HD inline float ssm2_from_bf16(ssm2_bf16 v) {
  return ssm2_bits_float((uint32_t)v.bits << 16);
}

SSM2_HD inline float ssm2_round_bf16(float v) {
  return ssm2_from_bf16(ssm2_to_bf16(v));
}

// Keep TF32's 10 explicit mantissa bits, rounding to nearest with ties away
// from zero like cvt.rna.tf32.f32; Inf and NaN pass through.
SSM2_HD inline float ssm2_round_tf32(float v) {
  const uint32_t u = ssm2_float_bits(v);
  if ((u & 0x7f800000u) == 0x7f800000u)
    return v;
  return ssm2_bits_float((u + 0x1000u) & 0xffffe000u);
}

// Load four contiguous BF16 values through one 8-byte access.
SSM2_HD inline void ssm2_ld4_bf16(const ssm2_bf16 *p, float v[4]) {
  const ssm2_bf16x4 q = *reinterpret_cast<const ssm2_bf16x4 *>(p);
  v[0] = ssm2_from_bf16(q.x);
  v[1] = ssm2_from_bf16(q.y);
  v[2] = ssm2_from_bf16(q.z);
  v[3] = ssm2_from_bf16(q.w);
}

// Round four values to BF16 and store them through one 8-byte access.
SSM2_HD inline void ssm2_st4_bf16(ssm2_bf16 *p, float a, float b, float c,
                                  float d) {
  const ssm2_bf16x4 q = {ssm2_to_bf16(a), ssm2_to_bf16(b), ssm2_to_bf16(c),
                         ssm2_to_bf16(d)};
  *reinterpret_cast<ssm2_bf16x4 *>(p) = q;
}
