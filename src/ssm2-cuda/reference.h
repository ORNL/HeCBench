// Scalar Mamba-2 SSD recurrence used as the correctness oracle.
//
// For head h, head channel p, and state channel n:
//   state_t[p,n] = exp(a[h] * dt_t[h]) * state_{t-1}[p,n]
//                  + x_t[h,p] * dt_t[h] * b_t[group(h),n]
//   out_t[h,p]   = dskip[h] * x_t[h,p]
//                  + sum_n c_t[group(h),n] * state_t[p,n]
//
// B and C are shared by all heads in a group; each (batch, head) recurrence is
// otherwise independent. The CUDA implementation algebraically chunks this
// same recurrence into cumulative decay, state propagation, and output kernels.
//
// Like the TileGym test reference, the oracle runs in FP32 on the BF16-rounded
// inputs and keeps its state in FP32.

#pragma once

#include <math.h>
#include <stddef.h>
#include <string.h>

// The device compiler parses this header for the index helpers. OpenMP is only
// used by the host reference.
#if !defined(__SYCL_DEVICE_ONLY__)
#include <omp.h>
#endif

#include "ssm2_types.h"

// Flatten x/out coordinates for the [B, T, H, P] row-major layout.
SSM2_HD inline size_t ssm2_x_idx(size_t b, size_t t, size_t h, size_t p,
                                 size_t T, size_t H, size_t P) {
  return (((b * T + t) * H + h) * P) + p;
}

// Flatten B/C coordinates for the [B, T, G, N] row-major layout.
SSM2_HD inline size_t ssm2_bc_idx(size_t b, size_t t, size_t g, size_t n,
                                  size_t T, size_t G, size_t N) {
  return (((b * T + t) * G + g) * N) + n;
}

// Flatten dt/decay coordinates for the [B, T, H] row-major layout.
SSM2_HD inline size_t ssm2_dt_idx(size_t b, size_t t, size_t h, size_t T,
                                  size_t H) {
  return (b * T + t) * H + h;
}

// Flatten recurrent-state coordinates for the [B, H, P, N] layout.
SSM2_HD inline size_t ssm2_st_idx(size_t b, size_t h, size_t p, size_t n,
                                  size_t H, size_t P, size_t N) {
  return (((b * H + h) * P + p) * N) + n;
}

// Flatten chunk-entry states for the [B, H, chunks, P, N] layout.
SSM2_HD inline size_t ssm2_hid_idx(size_t b, size_t h, size_t chunk, size_t p,
                                   size_t n, size_t H, size_t nchunks, size_t P,
                                   size_t N) {
  return (((((b * H + h) * nchunks + chunk) * P) + p) * N) + n;
}

// Evaluate the recurrence in time order and retain the state after the last
// token. Parallel tasks own disjoint [p0, p1) slices of one batch/head chain,
// so they can update final_state and out without synchronization.
// The SYCL device pass parses this translation unit, so it needs a declaration
// and does not run the OpenMP reference.
#if defined(__SYCL_DEVICE_ONLY__)
inline int omp_get_max_threads() { return 1; }
inline void ssm2_reference(const float *, const float *, const float *,
                           const float *, const float *, const float *,
                           const float *, float *, float *, size_t, size_t,
                           size_t, size_t, size_t, size_t) {}
#else
inline void ssm2_reference(const float *x, const float *a, const float *b,
                           const float *c, const float *dskip, const float *dt,
                           const float *init_state, float *out,
                           float *final_state, size_t B, size_t T, size_t H,
                           size_t G, size_t P, size_t N) {
  const size_t heads_per_group = H / G;
  const size_t state_elems = B * H * P * N;
  memcpy(final_state, init_state, state_elems * sizeof(float));

  const size_t heads = B * H;
  size_t splits = 1;
  const size_t threads = (size_t)omp_get_max_threads();
  // A batch/head chain is sequential in T. Split its independent P dimension
  // only when there are too few chains to occupy the available host threads.
  if (threads > 1 && heads > 0 && heads < threads) {
    splits = (threads + heads - 1) / heads;
    if (splits > P)
      splits = P;
    if (splits < 1)
      splits = 1;
  }
  const size_t tasks = heads * splits;

  // Run one P slice of one batch/head recurrence from t=0 through T-1.
  auto run_task = [&](size_t task) {
    const size_t chain = task / splits;
    const size_t split = task - chain * splits;
    const size_t batch = chain / H;
    const size_t h = chain - batch * H;
    const size_t group = h / heads_per_group;
    const size_t p0 = (split * P) / splits;
    const size_t p1 = ((split + 1) * P) / splits;
    const float ah = a[h];
    const float skip = dskip[h];
    for (size_t t = 0; t < T; ++t) {
      const float dtv = dt[ssm2_dt_idx(batch, t, h, T, H)];
      const float decay = expf(ah * dtv);
      const float *bv = b + ssm2_bc_idx(batch, t, group, 0, T, G, N);
      const float *cv = c + ssm2_bc_idx(batch, t, group, 0, T, G, N);
      for (size_t p = p0; p < p1; ++p) {
        const float xv = x[ssm2_x_idx(batch, t, h, p, T, H, P)];
        const float xdt = xv * dtv;
        float *hs = final_state + ssm2_st_idx(batch, h, p, 0, H, P, N);
        float y = skip * xv;
        for (size_t n = 0; n < N; ++n) {
          const float hsn = decay * hs[n] + xdt * bv[n];
          hs[n] = hsn;
          y += cv[n] * hsn;
        }
        out[ssm2_x_idx(batch, t, h, p, T, H, P)] = y;
      }
    }
  };

#pragma omp parallel for schedule(static)
  for (size_t task = 0; task < tasks; ++task)
    run_task(task);
}
#endif
