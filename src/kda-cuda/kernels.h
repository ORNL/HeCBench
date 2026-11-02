#ifndef KDA_KERNELS_H
#define KDA_KERNELS_H

#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_bf16.h>
#include "kda_tune.h"

// One block per (batch, value-head, V-tile). Several tokens are staged
// together so the serial recurrence hits a block barrier once per tile, and
// PARTS consecutive lanes share a value column with XOR shuffles.
constexpr int kda_warp_size = 32;

template <int NK, int NV>
kda_launch_plan kda_plan(int B, int HV, int sm_count, int warp = 32)
{
  return kda::make_plan<NK, NV>(B, HV, sm_count, warp);
}

template <int NK, int NV, int PARTS, int VTILE, int UNROLL>
__global__ void fused_recurrent_kda_kernel(
    const nv_bfloat16* __restrict__ q,     // [B,T,H,K]  queries
    const nv_bfloat16* __restrict__ k,     // [B,T,H,K]  keys
    const nv_bfloat16* __restrict__ v,     // [B,T,HV,V] values
    const float* __restrict__ g,           // [B,T,HV,K] log-space decay gates
    const nv_bfloat16* __restrict__ beta,  // [B,T,HV]   delta-rule step sizes
    nv_bfloat16* __restrict__ o,           // [B,T,HV,V] outputs
    float* __restrict__ h_final,           // [B,HV,K,V] optional final state
    const float scale,                     // 1/sqrt(K) applied to q
    const int B, const int T, const int T_stride, const int H, const int HV,
    const int V, const int G)
{
  (void)B;
  static_assert(PARTS > 0 && PARTS <= 32 && (PARTS & (PARTS - 1)) == 0,
                "PARTS must be a power of two no larger than a warp");
  static_assert(NK % PARTS == 0, "NK must be divisible by PARTS");
  static_assert(NV % VTILE == 0, "NV must be divisible by VTILE");
  static_assert(UNROLL >= 1 && UNROLL <= 8, "UNROLL must be in [1, 8]");
  static_assert((VTILE * PARTS) % kda_warp_size == 0 && VTILE * PARTS <= 1024,
                "Block size must be whole warps and at most 1024 threads");

  constexpr int NKP = NK / PARTS;   // K-rows of S this thread owns
  constexpr int TILES = NV / VTILE; // V-tiles that cover one value head

  const int bh = blockIdx.x / TILES;           // packed (batch, value-head)
  const int tile = blockIdx.x - bh * TILES;    // which V-tile this block owns
  const int b = bh / HV;                       // batch index
  const int hv = bh - b * HV;                  // value-head index
  const int h = hv / G;                        // matching query/key head
  const int vd_local = threadIdx.x / PARTS;    // column inside this V-tile
  const int vd = tile * VTILE + vd_local;      // global value-dimension index
  const int part = threadIdx.x - vd_local * PARTS; // which K-slice of that column
  const int tid = threadIdx.x;                 // flat thread id in the block

  float s[NKP];                                // this thread's slice of one S column
#pragma unroll
  for (int i = 0; i < NKP; i++) s[i] = 0.f;    // S starts at zero

  __shared__ float sh_q[UNROLL][NK];           // staged queries (already scaled)
  __shared__ float sh_k[UNROLL][NK];           // staged keys
  __shared__ float sh_d[UNROLL][NK];           // staged exp(g) decay
  __shared__ float sh_v[UNROLL][VTILE];        // staged values for this V-tile
  __shared__ float sh_beta[UNROLL];            // staged delta-rule betas

  const size_t h_off = (size_t)h * NK;                // q/k offset of head h
  const size_t hvk_off = (size_t)hv * NK;             // g offset of value head hv
  const size_t hvv_off = (size_t)hv * V + tile * VTILE; // v/o offset of this V-tile

  for (int t0 = 0; t0 < T; t0 += UNROLL) {            // walk the sequence in token tiles
    const int nstep = (T - t0 >= UNROLL) ? UNROLL : (T - t0); // tokens in this tile
    const int nqk = nstep * NK;                       // q/k/g elements to stage
    const int nvload = nstep * VTILE;                 // v elements to stage

    for (int i = tid; i < nqk; i += blockDim.x) {     // cooperative load of q, k, g
      const int sidx = i / NK;                        // token within the tile
      const int d = i - sidx * NK;                    // K-dimension index
      const size_t tb = (size_t)b * T_stride + t0 + sidx; // (batch, time) packed index
      sh_q[sidx][d] =                                 // q * scale, bf16 -> fp32
          __bfloat162float(q[tb * H * NK + h_off + d]) * scale;
      sh_k[sidx][d] = __bfloat162float(k[tb * H * NK + h_off + d]); // k, bf16 -> fp32
      sh_d[sidx][d] = __expf(g[tb * HV * NK + hvk_off + d]);        // decay = exp(g)
    }
    for (int i = tid; i < nvload; i += blockDim.x) {  // cooperative load of this V-tile of v
      const int sidx = i / VTILE;                     // token within the tile
      const int d = i - sidx * VTILE;                 // local V-dimension
      const size_t tb = (size_t)b * T_stride + t0 + sidx;
      sh_v[sidx][d] = __bfloat162float(               // v, bf16 -> fp32
          v[tb * HV * V + hvv_off + d]);
    }
    if (tid < nstep) {                                // one thread per token loads beta
      const size_t tb = (size_t)b * T_stride + t0 + tid;
      sh_beta[tid] = __bfloat162float(beta[tb * HV + hv]);
    }
    __syncthreads();                                  // all staged data visible to every thread

#pragma unroll
    for (int sidx = 0; sidx < UNROLL; sidx++) {       // serial scan over tokens in the tile
      if (sidx >= nstep) break;                       // last tile may be shorter
      float prediction = 0.f;                         // partial k · S for this column
#pragma unroll
      for (int i = 0; i < NKP; i++) {                 // each thread owns NKP rows of S
        const int kk = i * PARTS + part;              // global K-index for this slice
        s[i] *= sh_d[sidx][kk];                       // decay this row: S[k,:] *= exp(g[k])
        prediction = fmaf(sh_k[sidx][kk], s[i], prediction); // accumulate k[k] * S[k,vd]
      }
#pragma unroll
      for (int offset = PARTS / 2; offset > 0; offset >>= 1)
        prediction += __shfl_xor_sync(0xffffffff, prediction, offset); // sum PARTS lanes
      const float u = sh_v[sidx][vd_local] - prediction; // residual u = v - k·S
      const float bb = sh_beta[sidx];                    // delta-rule learning rate
      float acc = 0.f;                                   // partial q · S after the write
#pragma unroll
      for (int i = 0; i < NKP; i++) {                    // write residual, then form output
        const int kk = i * PARTS + part;                 // same K-index as above
        s[i] = fmaf(bb * sh_k[sidx][kk], u, s[i]);       // S[k,vd] += (beta*k[k]) * u[vd]
        acc = fmaf(sh_q[sidx][kk], s[i], acc);           // accumulate q[k] * S[k,vd]
      }
#pragma unroll
      for (int offset = PARTS / 2; offset > 0; offset >>= 1)
        acc += __shfl_xor_sync(0xffffffff, acc, offset); // sum PARTS lanes -> full q·S
      if (part == 0) {                                   // one writer per value column
        const size_t tb = (size_t)b * T_stride + t0 + sidx;
        const size_t v_head = tb * HV * V + hvv_off; // output row for this token
        o[v_head + vd_local] = __float2bfloat16(acc);       // store o[vd] as bf16
      }
    }
    __syncthreads();                                     // reuse shared buffers for next tile
  }

  if (h_final != nullptr) {                              // optional dump of the recurrent state
#pragma unroll
    for (int i = 0; i < NKP; i++) {
      const int kk = i * PARTS + part;                   // global K-index of this slice
      h_final[(((size_t)b * HV + hv) * NK + kk) * V + vd] = s[i];
    }
  }
}

template <int NK, int NV, int PARTS, int VT>
void kda_dispatch(
    const nv_bfloat16* q, const nv_bfloat16* k, const nv_bfloat16* v,
    const float* g, const nv_bfloat16* beta, nv_bfloat16* o, float* h_final,
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
    const nv_bfloat16* q, const nv_bfloat16* k, const nv_bfloat16* v,
    const float* g, const nv_bfloat16* beta, nv_bfloat16* o, float* h_final,
    float scale, int B, int T, int H, int HV, int V, int G, int vtile,
    int T_stride = -1)
{
  if (T_stride <= 0) T_stride = T;
  constexpr int PARTS = kda::parts_for<NK>();
  kda_dispatch<NK, NV, PARTS, kda::max_vtile(NV, PARTS, kda::kDispatchWarp)>(
      q, k, v, g, beta, o, h_final, scale, B, T, T_stride, H, HV, V, G, vtile);
}

#endif
