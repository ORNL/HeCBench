#ifndef KDA_KERNELS_H
#define KDA_KERNELS_H

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <sycl/sycl.hpp>
#include "kda_tune.h"

using kda_bf16 = sycl::ext::oneapi::bfloat16;

constexpr int kda_warp_align = 32;

inline int kda_device_wave(const sycl::device& dev)
{
  const auto sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
  auto r = std::max_element(sizes.begin(), sizes.end());
  return *r;
}

template <int NK, int NV>
kda_launch_plan kda_plan(int B, int HV, int sm_count, int warp)
{
  return kda::make_plan<NK, NV>(B, HV, sm_count, warp);
}

template <int NK, int NV, int PARTS, int VTILE, int UNROLL>
void fused_recurrent_kda_kernel(
    sycl::nd_item<1> item,
    const kda_bf16* q, const kda_bf16* k, const kda_bf16* v,
    const float* g, const kda_bf16* beta, kda_bf16* o, float* h_final,
    sycl::local_accessor<float, 1> smem,
    float scale, int B, int T, int T_stride, int H, int HV, int V, int G)
{
  (void)B;
  static_assert(PARTS > 0 && PARTS <= 32 && (PARTS & (PARTS - 1)) == 0,
                "PARTS must be a power of two no larger than a sub-group");
  static_assert(NK % PARTS == 0, "NK must be divisible by PARTS");
  static_assert(NV % VTILE == 0, "NV must be divisible by VTILE");
  static_assert(UNROLL >= 1 && UNROLL <= 8, "UNROLL must be in [1, 8]");
  static_assert((VTILE * PARTS) % kda_warp_align == 0 && VTILE * PARTS <= 1024,
                "Block size must be a multiple of 32 and at most 1024 threads");

  constexpr int NKP = NK / PARTS;
  constexpr int TILES = NV / VTILE;
  constexpr int QOFF = 0;
  constexpr int KOFF = UNROLL * NK;
  constexpr int DOFF = 2 * UNROLL * NK;
  constexpr int VOFF = 3 * UNROLL * NK;
  constexpr int BOFF = VOFF + UNROLL * VTILE;

  const int tid = item.get_local_id(0);
  const int bdim = item.get_local_range(0);
  const int bh = item.get_group(0) / TILES;
  const int tile = item.get_group(0) - bh * TILES;
  const int b = bh / HV;
  const int hv = bh - b * HV;
  const int h = hv / G;
  const int vd_local = tid / PARTS;
  const int vd = tile * VTILE + vd_local;
  const int part = tid - vd_local * PARTS;
  auto grp = item.get_group();
  auto sg = item.get_sub_group();

  float s[NKP];
#pragma unroll
  for (int i = 0; i < NKP; i++) s[i] = 0.f;

  const size_t h_off = (size_t)h * NK;
  const size_t hvk_off = (size_t)hv * NK;
  const size_t hvv_off = (size_t)hv * V + tile * VTILE;

  for (int t0 = 0; t0 < T; t0 += UNROLL) {
    const int nstep = (T - t0 >= UNROLL) ? UNROLL : (T - t0);
    const int nqk = nstep * NK;
    const int nvload = nstep * VTILE;

    for (int i = tid; i < nqk; i += bdim) {
      const int sidx = i / NK;
      const int d = i - sidx * NK;
      const size_t tb = (size_t)b * T_stride + t0 + sidx;
      smem[QOFF + sidx * NK + d] =
          float(q[tb * H * NK + h_off + d]) * scale;
      smem[KOFF + sidx * NK + d] =
          float(k[tb * H * NK + h_off + d]);
      smem[DOFF + sidx * NK + d] =
          sycl::native::exp(g[tb * HV * NK + hvk_off + d]);
    }
    for (int i = tid; i < nvload; i += bdim) {
      const int sidx = i / VTILE;
      const int d = i - sidx * VTILE;
      const size_t tb = (size_t)b * T_stride + t0 + sidx;
      smem[VOFF + sidx * VTILE + d] = float(
          v[tb * HV * V + hvv_off + d]);
    }
    if (tid < nstep) {
      const size_t tb = (size_t)b * T_stride + t0 + tid;
      smem[BOFF + tid] = float(beta[tb * HV + hv]);
    }
    sycl::group_barrier(grp);

#pragma unroll
    for (int sidx = 0; sidx < UNROLL; sidx++) {
      if (sidx >= nstep) break;
      float prediction = 0.f;
#pragma unroll
      for (int i = 0; i < NKP; i++) {
        const int kk = i * PARTS + part;
        s[i] *= smem[DOFF + sidx * NK + kk];
        prediction = sycl::fma(smem[KOFF + sidx * NK + kk], s[i], prediction);
      }
#pragma unroll
      for (int offset = PARTS / 2; offset > 0; offset >>= 1)
        prediction += sycl::permute_group_by_xor(sg, prediction, offset);
      const float u = smem[VOFF + sidx * VTILE + vd_local] - prediction;
      const float bb = smem[BOFF + sidx];
      float acc = 0.f;
#pragma unroll
      for (int i = 0; i < NKP; i++) {
        const int kk = i * PARTS + part;
        s[i] = sycl::fma(bb * smem[KOFF + sidx * NK + kk], u, s[i]);
        acc = sycl::fma(smem[QOFF + sidx * NK + kk], s[i], acc);
      }
#pragma unroll
      for (int offset = PARTS / 2; offset > 0; offset >>= 1)
        acc += sycl::permute_group_by_xor(sg, acc, offset);
      if (part == 0) {
        const size_t tb = (size_t)b * T_stride + t0 + sidx;
        const size_t v_head = tb * HV * V + hvv_off;
        o[v_head + vd_local] = kda_bf16(acc);
      }
    }
    sycl::group_barrier(grp);
  }

  if (h_final != nullptr) {
#pragma unroll
    for (int i = 0; i < NKP; i++) {
      const int kk = i * PARTS + part;
      h_final[(((size_t)b * HV + hv) * NK + kk) * V + vd] = s[i];
    }
  }
}

template <int NK, int NV, int PARTS, int VT, int SG>
void kda_dispatch(
    sycl::queue& q,
    const kda_bf16* qq, const kda_bf16* k, const kda_bf16* v,
    const float* g, const kda_bf16* beta, kda_bf16* o, float* h_final,
    float scale, int B, int T, int T_stride, int H, int HV, int V, int G, int vtile)
{
  static_assert(SG == 32 || SG == 64, "sub-group size must be 32 or 64");
  if (vtile == VT) {
    constexpr int U = kda::unroll_for<NK, VT>();
    const int grid = B * HV * (NV / VT);
    const int block = VT * PARTS;
    q.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> smem(
          sycl::range<1>((size_t)U * (3 * NK + VT + 1)), cgh);
      cgh.parallel_for(
          sycl::nd_range<1>((size_t)grid * block, (size_t)block),
          [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(SG)]] {
            fused_recurrent_kda_kernel<NK, NV, PARTS, VT, U>(
                item, qq, k, v, g, beta, o, h_final, smem,
                scale, B, T, T_stride, H, HV, V, G);
          });
    });
    return;
  }
  if constexpr (kda::can_halve(NV, VT, PARTS, kda::kDispatchWarp))
    kda_dispatch<NK, NV, PARTS, VT / 2, SG>(
        q, qq, k, v, g, beta, o, h_final, scale, B, T, T_stride, H, HV, V, G, vtile);
  else {
    std::fprintf(stderr, "Error: no fused KDA kernel for vtile=%d\n", vtile);
    std::exit(1);
  }
}

template <int NK, int NV>
void kda_launch(
    sycl::queue& q,
    const kda_bf16* qq, const kda_bf16* k, const kda_bf16* v,
    const float* g, const kda_bf16* beta, kda_bf16* o, float* h_final,
    float scale, int B, int T, int H, int HV, int V, int G, int vtile, int wave,
    int T_stride = -1)
{
  if (T_stride <= 0) T_stride = T;
  constexpr int PARTS = kda::parts_for<NK>();
  constexpr int VTMAX = kda::max_vtile(NV, PARTS, kda::kDispatchWarp);
  if (wave == 64)
    kda_dispatch<NK, NV, PARTS, VTMAX, 64>(
        q, qq, k, v, g, beta, o, h_final, scale, B, T, T_stride, H, HV, V, G, vtile);
  else
    kda_dispatch<NK, NV, PARTS, VTMAX, 32>(
        q, qq, k, v, g, beta, o, h_final, scale, B, T, T_stride, H, HV, V, G, vtile);
}

#endif
