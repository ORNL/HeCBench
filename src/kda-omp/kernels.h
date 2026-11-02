#ifndef KDA_KERNELS_H
#define KDA_KERNELS_H

#include <cmath>
#include <cstddef>

// Arithmetic stays fp32 after widening
using kda_f16 = _Float16;

inline void pack_f16(const float* src, kda_f16* dst, size_t count)
{
  for (size_t i = 0; i < count; ++i)
    dst[i] = (kda_f16)src[i];
}

inline void unpack_f16(const kda_f16* src, float* dst, size_t count)
{
  for (size_t i = 0; i < count; ++i)
    dst[i] = (float)src[i];
}

// Fused recurrent forward kernel for KDA (Kimi Delta Attention), based on
//   https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/fused_recurrent.py
//
// Parallelization (mirrors the CUDA version):
//   - One team per (batch, value-head) pair  => num_teams = B * HV
//   - One thread per value dimension         => thread_limit = V
//   - Each thread keeps its own state column S[:, vd] (length K) in registers.
//
// The CUDA kernel additionally stages the per-time-step q/k/exp(g) vectors in
// shared memory so the K exp() evaluations are shared by the V threads of the
// block. That is not reproduced here: it needs the team to have exactly V
// threads (a barrier per step, indexed by thread id), while OpenMP only lets us
// request thread_limit(V) and may hand out fewer threads. The inner loop is
// therefore written as a plain `parallel for` over the value dimension, which
// keeps the state column per-thread at the cost of re-evaluating exp(g).
//
// NK is the head (key) dimension as a compile-time constant so the inner loops
// unroll and the state column stays in registers.
template <int NK>
void fused_recurrent_kda(
    const kda_f16* q,     // [B, T_stride, H,  NK]
    const kda_f16* k,     // [B, T_stride, H,  NK]
    const kda_f16* v,     // [B, T_stride, HV, V]
    const float* g,       // [B, T_stride, HV, NK]
    const kda_f16* beta,  // [B, T_stride, HV]
    kda_f16* o,           // [B, T_stride, HV, V]
    float* h_final,       // [B, HV, NK, V]
    // The CUDA kernel skips the final-state write by passing a null pointer;
    // h_final stays mapped here, so the write is gated by this flag instead.
    const int store_h_final,
    const float scale,
    const int B, const int T, const int H, const int HV,
    const int V, const int G,
    const int T_stride = -1)
{
  const int t_stride = T_stride > 0 ? T_stride : T;
  const int BH = B * HV;

  #pragma omp target teams distribute num_teams(BH) thread_limit(V)
  for (int bh = 0; bh < BH; bh++) {
    const int b  = bh / HV;
    const int hv = bh % HV;
    const int h  = hv / G;

    #pragma omp parallel for
    for (int vd = 0; vd < V; vd++) {

      float s[NK];
      for (int i = 0; i < NK; i++) s[i] = 0.f;

      for (int t = 0; t < T; t++) {
        const size_t tb = (size_t)b * t_stride + t;
        const size_t qk_base = (tb * H  + h)  * NK;
        const size_t g_base  = (tb * HV + hv) * NK;
        const size_t v_base  = (tb * HV + hv) * V + vd;

        float u = (float)v[v_base];

        for (int i = 0; i < NK; i++) {
          s[i] *= expf(g[g_base + i]);
          u -= (float)k[qk_base + i] * s[i];
        }

        const float bb = (float)beta[tb * HV + hv];
        float acc = 0.f;
        for (int i = 0; i < NK; i++) {
          s[i] += bb * (float)k[qk_base + i] * u;
          acc += ((float)q[qk_base + i] * scale) * s[i];
        }
        o[v_base] = (kda_f16)acc;
      }

      if (store_h_final) {
        for (int i = 0; i < NK; i++)
          h_final[(((size_t)b * HV + hv) * NK + i) * V + vd] = s[i];
      }
    }
  }
}

#endif  // KDA_KERNELS_H
