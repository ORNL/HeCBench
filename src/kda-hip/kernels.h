#ifndef KDA_KERNELS_H
#define KDA_KERNELS_H

#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#include "kda_tune.h"

// Portable fused recurrent KDA: SIMT HIP with wavefront shuffles and static
// shared memory. Wave size is taken from the device (32 on NVIDIA HIP, 64 on
// AMD). Tile dispatch is instantiated at wave 32; runtime planning keeps
// blocks that fill the native wave.
constexpr int kda_warp_align = 32;

template <int NK, int NV>
kda_launch_plan kda_plan(int B, int HV, int sm_count, int warp)
{
  return kda::make_plan<NK, NV>(B, HV, sm_count, warp);
}

template <int NK, int NV, int PARTS, int VTILE, int UNROLL>
__global__ void fused_recurrent_kda_kernel(
    const hip_bfloat16* __restrict__ q,
    const hip_bfloat16* __restrict__ k,
    const hip_bfloat16* __restrict__ v,
    const float* __restrict__ g,
    const hip_bfloat16* __restrict__ beta,
    hip_bfloat16* __restrict__ o,
    float* __restrict__ h_final,
    const float scale,
    const int B, const int T, const int T_stride, const int H, const int HV,
    const int V, const int G)
{
  (void)B;
  static_assert(PARTS > 0 && PARTS <= 32 && (PARTS & (PARTS - 1)) == 0,
                "PARTS must be a power of two no larger than 32");
  static_assert(NK % PARTS == 0, "NK must be divisible by PARTS");
  static_assert(NV % VTILE == 0, "NV must be divisible by VTILE");
  static_assert(UNROLL >= 1 && UNROLL <= 8, "UNROLL must be in [1, 8]");
  static_assert((VTILE * PARTS) % kda_warp_align == 0 && VTILE * PARTS <= 1024,
                "Block size must be a multiple of 32 and at most 1024 threads");

  constexpr int NKP = NK / PARTS;
  constexpr int TILES = NV / VTILE;

  const int bh = blockIdx.x / TILES;
  const int tile = blockIdx.x - bh * TILES;
  const int b = bh / HV;
  const int hv = bh - b * HV;
  const int h = hv / G;
  const int vd_local = threadIdx.x / PARTS;
  const int vd = tile * VTILE + vd_local;
  const int part = threadIdx.x - vd_local * PARTS;
  const int tid = threadIdx.x;

  float s[NKP];
#pragma unroll
  for (int i = 0; i < NKP; i++) s[i] = 0.f;

  __shared__ float sh_q[UNROLL][NK];
  __shared__ float sh_k[UNROLL][NK];
  __shared__ float sh_d[UNROLL][NK];
  __shared__ float sh_v[UNROLL][VTILE];
  __shared__ float sh_beta[UNROLL];

  const size_t h_off = (size_t)h * NK;
  const size_t hvk_off = (size_t)hv * NK;
  const size_t hvv_off = (size_t)hv * V + tile * VTILE;

  for (int t0 = 0; t0 < T; t0 += UNROLL) {
    const int nstep = (T - t0 >= UNROLL) ? UNROLL : (T - t0);
    const int nqk = nstep * NK;
    const int nvload = nstep * VTILE;

    for (int i = tid; i < nqk; i += (int)blockDim.x) {
      const int sidx = i / NK;
      const int d = i - sidx * NK;
      const size_t tb = (size_t)b * T_stride + t0 + sidx;
      sh_q[sidx][d] =
          (float)(q[tb * H * NK + h_off + d]) * scale;
      sh_k[sidx][d] = (float)(k[tb * H * NK + h_off + d]);
      sh_d[sidx][d] = __expf(g[tb * HV * NK + hvk_off + d]);
    }
    for (int i = tid; i < nvload; i += (int)blockDim.x) {
      const int sidx = i / VTILE;
      const int d = i - sidx * VTILE;
      const size_t tb = (size_t)b * T_stride + t0 + sidx;
      sh_v[sidx][d] = (float)(
          v[tb * HV * V + hvv_off + d]);
    }
    if (tid < nstep) {
      const size_t tb = (size_t)b * T_stride + t0 + tid;
      sh_beta[tid] = (float)(beta[tb * HV + hv]);
    }
    __syncthreads();

#pragma unroll
    for (int sidx = 0; sidx < UNROLL; sidx++) {
      if (sidx >= nstep) break;
      float prediction = 0.f;
#pragma unroll
      for (int i = 0; i < NKP; i++) {
        const int kk = i * PARTS + part;
        s[i] *= sh_d[sidx][kk];
        prediction = fmaf(sh_k[sidx][kk], s[i], prediction);
      }
#pragma unroll
      for (int offset = PARTS / 2; offset > 0; offset >>= 1)
        prediction += __shfl_xor(prediction, offset);
      const float u = sh_v[sidx][vd_local] - prediction;
      const float bb = sh_beta[sidx];
      float acc = 0.f;
#pragma unroll
      for (int i = 0; i < NKP; i++) {
        const int kk = i * PARTS + part;
        s[i] = fmaf(bb * sh_k[sidx][kk], u, s[i]);
        acc = fmaf(sh_q[sidx][kk], s[i], acc);
      }
#pragma unroll
      for (int offset = PARTS / 2; offset > 0; offset >>= 1)
        acc += __shfl_xor(acc, offset);
      if (part == 0) {
        const size_t tb = (size_t)b * T_stride + t0 + sidx;
        const size_t v_head = tb * HV * V + hvv_off;
        o[v_head + vd_local] = hip_bfloat16(acc);
      }
    }
    __syncthreads();
  }

  if (h_final != nullptr) {
#pragma unroll
    for (int i = 0; i < NKP; i++) {
      const int kk = i * PARTS + part;
      h_final[(((size_t)b * HV + hv) * NK + kk) * V + vd] = s[i];
    }
  }
}

template <int NK, int NV, int PARTS, int VT>
void kda_dispatch(
    const hip_bfloat16* q, const hip_bfloat16* k, const hip_bfloat16* v,
    const float* g, const hip_bfloat16* beta, hip_bfloat16* o, float* h_final,
    float scale, int B, int T, int T_stride, int H, int HV, int V, int G, int vtile)
{
  if (vtile == VT) {
    constexpr int U = kda::unroll_for<NK, VT>();
    const dim3 grid(B * HV * (NV / VT));
    const dim3 block(VT * PARTS);
    fused_recurrent_kda_kernel<NK, NV, PARTS, VT, U><<<grid, block>>>(
        q, k, v, g, beta, o, h_final, scale, B, T, T_stride, H, HV, V, G);
    return;
  }
  if constexpr (kda::can_halve(NV, VT, PARTS, kda::kDispatchWarp))
    kda_dispatch<NK, NV, PARTS, VT / 2>(q, k, v, g, beta, o, h_final, scale,
                                        B, T, T_stride, H, HV, V, G, vtile);
  else {
    std::fprintf(stderr, "Error: no fused KDA kernel for vtile=%d\n", vtile);
    std::exit(1);
  }
}

template <int NK, int NV>
void kda_launch(
    const hip_bfloat16* q, const hip_bfloat16* k, const hip_bfloat16* v,
    const float* g, const hip_bfloat16* beta, hip_bfloat16* o, float* h_final,
    float scale, int B, int T, int H, int HV, int V, int G, int vtile,
    int T_stride = -1)
{
  if (T_stride <= 0) T_stride = T;
  constexpr int PARTS = kda::parts_for<NK>();
  kda_dispatch<NK, NV, PARTS, kda::max_vtile(NV, PARTS, kda::kDispatchWarp)>(
      q, k, v, g, beta, o, h_final, scale, B, T, T_stride, H, HV, V, G, vtile);
}

#endif
