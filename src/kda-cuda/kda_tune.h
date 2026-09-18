#ifndef KDA_TUNE_H
#define KDA_TUNE_H

#include <cstddef>

// Compile-time launch-shape helpers shared by the CUDA, HIP, and SYCL fused
// recurrent KDA kernels. Tile dispatch is instantiated for wave/sub-group 32
// (the finer grid). Runtime planning then keeps only tiles whose block size
// is a multiple of the device wave size, which is 32 or 64.
namespace kda {

constexpr int floor_pow2(int x)
{
  int p = 1;
  while (p * 2 <= x) p *= 2;
  return p;
}

// Threads per state column. Roughly four state elements per thread keeps the
// unrolled inner loops in registers without spilling on any supported arch.
constexpr int state_parts(int K)
{
  int p = floor_pow2(K >= 4 ? K / 4 : 1);
  if (p > 32) p = 32;
  while (p > 1 && K % p != 0) p /= 2;
  return p;
}

// Smallest V-tile whose block is a whole number of warps/wavefronts.
constexpr int min_vtile(int parts, int warp)
{
  int t = 1;
  while ((t * parts) % warp != 0) t *= 2;
  return t;
}

// Largest V-tile that divides V and keeps the block within 1024 threads.
constexpr int max_vtile(int V, int parts, int warp)
{
  int t = floor_pow2(1024 / parts);
  if (t > V) t = floor_pow2(V);
  while (t > 1 && V % t != 0) t /= 2;
  const int lo = min_vtile(parts, warp);
  return t < lo ? lo : t;
}

constexpr bool can_halve(int V, int vtile, int parts, int warp)
{
  const int next = vtile / 2;
  return next >= min_vtile(parts, warp) &&
         V % (next > 0 ? next : 1) == 0;
}

// Tokens staged per barrier. Capped by a shared-memory budget so several
// blocks stay resident per SM even on the 64 KB-per-SM architectures.
constexpr int token_tile(int K, int vtile)
{
  constexpr int budget = 16384;
  int u = 8;
  while (u > 1 && (3 * K + vtile) * u * (int)sizeof(float) > budget) u /= 2;
  return u;
}

template <int K>
constexpr int parts_for()
{
#ifdef KDA_STATE_PARTS
  return KDA_STATE_PARTS;
#else
  return state_parts(K);
#endif
}

template <int K, int VT>
constexpr int unroll_for()
{
#ifdef KDA_UNROLL
  return KDA_UNROLL;
#else
  return token_tile(K, VT);
#endif
}

struct launch_plan {
  int parts;
  int vtile;
  int unroll;
  int block;
  int grid;
  int warp;
};

inline bool wave_ok(int warp) { return warp == 32 || warp == 64; }

inline bool plan_ok(const launch_plan& plan)
{
  return plan.vtile > 0 && plan.block > 0 && plan.grid > 0 &&
         wave_ok(plan.warp) && (plan.block % plan.warp) == 0;
}

// Compile-time tile set covers wave 32 and 64: every 64-aligned block is
// also 32-aligned, so instantiating from warp 32 is enough.
constexpr int kDispatchWarp = 32;

// Pick the largest V-tile that still covers the device with blocks whose
// size is a multiple of the runtime wave/sub-group size (32 or 64).
template <int NK, int NV>
launch_plan make_plan(int B, int HV, int sm_count, int warp)
{
  constexpr int PARTS = parts_for<NK>();
  constexpr int VMAX = max_vtile(NV, PARTS, kDispatchWarp);
  constexpr int VMIN = min_vtile(PARTS, kDispatchWarp);

  int vtile = 0;
#ifdef KDA_VTILE
  // Accept the override only if the dispatcher instantiates it (halving from
  // VMAX) and the block is a multiple of the device wave.
  if (KDA_VTILE > 0 && NV % KDA_VTILE == 0 &&
      (KDA_VTILE * PARTS) % warp == 0 && KDA_VTILE * PARTS <= 1024) {
    for (int vt = VMAX;; vt /= 2) {
      if (vt == KDA_VTILE) {
        vtile = KDA_VTILE;
        break;
      }
      if (!can_halve(NV, vt, PARTS, kDispatchWarp)) break;
    }
  }
#else
  for (int vt = VMAX; vt >= VMIN; vt /= 2) {
    if (NV % vt != 0) continue;
    if ((vt * PARTS) % warp != 0) continue;
    vtile = vt;
    if ((size_t)B * HV * (NV / vt) >= (size_t)sm_count) break;
  }
#endif

  launch_plan plan{};
  plan.parts = PARTS;
  plan.vtile = vtile;
#ifdef KDA_UNROLL
  plan.unroll = (vtile > 0) ? KDA_UNROLL : 0;
#else
  plan.unroll = (vtile > 0) ? token_tile(NK, vtile) : 0;
#endif
  plan.block = vtile * PARTS;
  plan.grid = (vtile > 0) ? B * HV * (NV / vtile) : 0;
  plan.warp = warp;
  return plan;
}

}  // namespace kda

using kda_launch_plan = kda::launch_plan;

#endif
