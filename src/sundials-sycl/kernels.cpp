#include <chrono>
#include <cfloat>
#include <sycl/sycl.hpp>
#include "sundials.h"

#ifdef USE_GPU
#define SELECTOR sycl::gpu_selector_v
#else
#define SELECTOR sycl::cpu_selector_v
#endif

// Round the global range up to a multiple of the work-group size.
static inline sycl::range<1> gwsFor(size_t n, int bs) {
  return sycl::range<1>(((n + bs - 1) / bs) * (size_t)bs);
}

using clk = std::chrono::steady_clock;
static inline long ns_since(clk::time_point t0) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t0)
      .count();
}

/*
 * SYCL port of the SUNDIALS implicit-integration miniapp. The kernel structure
 * mirrors the CUDA/HIP version:
 *   * k_setup      fused ewtSS/SV + checkConstraints + CVDiag form/build/update
 *   * scaleAddI    out-of-place BCSR SUNMatScaleAddI (M = c*J + I)
 *   * matvecResid  fused BCSR matvec + nonlinear residual
 *   * reduce+adv   six sycl::reductions fused with the in-place state advance
 *   * scaleInv     renormalization by the device-resident max norm
 * An in-order queue keeps the whole step loop asynchronous (single wait at each
 * end); the reduction scalars stay in device memory and are copied back once.
 */
long sundials_miniapp(int steps, int newton, int bs, const sun_in *in,
                      sun_out *out) {
  const IDXT m = in->m, nblocks = in->nblocks, blocknnz = in->blocknnz;
  const size_t n         = (size_t)m * nblocks;
  const size_t nnz_total = (size_t)nblocks * blocknnz;

  sycl::queue q(SELECTOR, sycl::property::queue::in_order());

  // Device buffers (allocated once, resident across all steps).
  IDXT *d_rowptr = sycl::malloc_device<IDXT>(m + 1, q);
  IDXT *d_colind = sycl::malloc_device<IDXT>(blocknnz, q);
  real *d_ycur    = sycl::malloc_device<real>(n, q);
  real *d_Vabstol = sycl::malloc_device<real>(n, q);
  real *d_cflag   = sycl::malloc_device<real>(n, q);
  real *d_mm      = sycl::malloc_device<real>(n, q);
  real *d_zn1     = sycl::malloc_device<real>(n, q);
  real *d_ycor    = sycl::malloc_device<real>(n, q);
  real *d_fpred   = sycl::malloc_device<real>(n, q);
  real *d_ypred   = sycl::malloc_device<real>(n, q);
  real *d_Min     = sycl::malloc_device<real>(n, q);
  real *d_J       = sycl::malloc_device<real>(nnz_total, q);
  real *d_ewt     = sycl::malloc_device<real>(n, q);
  real *d_ewtsv   = sycl::malloc_device<real>(n, q);
  real *d_tempv   = sycl::malloc_device<real>(n, q);
  real *d_save    = sycl::malloc_device<real>(n, q);
  real *d_ftemp   = sycl::malloc_device<real>(n, q);
  real *d_yform   = sycl::malloc_device<real>(n, q);
  real *d_Mdiag   = sycl::malloc_device<real>(n, q);
  real *d_ybuild  = sycl::malloc_device<real>(n, q);
  real *d_Mupd    = sycl::malloc_device<real>(n, q);
  real *d_res     = sycl::malloc_device<real>(n, q);
  real *d_Mbcsr   = sycl::malloc_device<real>(nnz_total, q);
  real *d_Ax      = sycl::malloc_device<real>(n, q);
  // Reduction accumulators: 0 dot, 1 wL2, 2 max|res|, 3 L1, 4 min(res),
  // 5 max|ycur_new| (renorm).
  real *d_red = sycl::malloc_device<real>(6, q);

  q.memcpy(d_rowptr, in->rowptr, (m + 1) * sizeof(IDXT));
  q.memcpy(d_colind, in->colind, blocknnz * sizeof(IDXT));
  q.memcpy(d_ycur, in->ycur0, n * sizeof(real));
  q.memcpy(d_Vabstol, in->Vabstol, n * sizeof(real));
  q.memcpy(d_cflag, in->cflag, n * sizeof(real));
  q.memcpy(d_mm, in->mm, n * sizeof(real));
  q.memcpy(d_zn1, in->zn1, n * sizeof(real));
  q.memcpy(d_ycor, in->ycor, n * sizeof(real));
  q.memcpy(d_fpred, in->fpred, n * sizeof(real));
  q.memcpy(d_ypred, in->ypred, n * sizeof(real));
  q.memcpy(d_Min, in->Min, n * sizeof(real));
  q.memcpy(d_J, in->Jvals, nnz_total * sizeof(real));

  sycl::range<1> sgws = gwsFor(n, bs), slws(bs);
  int bcsr_bs = 128;
  int bcsr_gr = nblocks < 65535 ? nblocks : 65535;
  sycl::range<1> bgws((size_t)bcsr_gr * bcsr_bs), blws(bcsr_bs);

  const real reltol = in->reltol, Sabstol = in->Sabstol, rl1 = in->rl1,
             ngamma = in->ngamma, h = in->h, r = in->r, fract = in->fract,
             uround = in->uround, c = -in->gamma, damp = in->damp;

  q.wait();
  clk::time_point start = clk::now();

  for (int s = 0; s < steps; s++) {
    // ---- fused setup ----
    q.parallel_for(sycl::nd_range<1>(sgws, slws), [=](sycl::nd_item<1> item) {
      size_t i = item.get_global_id(0);
      if (i >= n) return;
      real yc     = d_ycur[i];
      real w      = 1.0 / (reltol * sycl::fabs(yc) + Sabstol);
      d_ewt[i]    = w;
      d_ewtsv[i]  = 1.0 / (reltol * sycl::fabs(yc) + d_Vabstol[i]);
      real tmp    = (sycl::fabs(d_cflag[i]) >= 1.5) ? 1.0 : 0.0;
      tmp         = tmp * d_cflag[i] / w;
      d_save[i]   = -0.1 * tmp;
      d_tempv[i]  = (yc - 0.1 * tmp) * d_mm[i];
      real ft     = h * d_fpred[i] - d_zn1[i];
      d_ftemp[i]  = ft;
      d_yform[i]  = r * ft + d_ypred[i];
      real mval    = fract * ft - h * (d_Min[i] - d_fpred[i]);
      real yv      = ft * w;
      bool test    = (sycl::fabs(yv) >= uround);
      real bit     = test ? 1.0 : 0.0;
      real bitcomp = test ? 0.0 : -1.0;
      yv           = fract * ft * bit - bitcomp;
      real md      = mval / yv * bit - bitcomp;
      d_Mdiag[i]   = md;
      d_ybuild[i]  = yv;
      d_Mupd[i]    = r * (1.0 / md - 1.0) + 1.0;
    });

    // ---- scaleAddI: form Newton matrix M = c*J + I (BCSR) ----
    q.parallel_for(sycl::nd_range<1>(bgws, blws), [=](sycl::nd_item<1> item) {
      const IDXT ngroups = item.get_group_range(0);
      const IDXT lsize   = item.get_local_range(0);
      for (IDXT block = item.get_group(0); block < nblocks; block += ngroups) {
        for (IDXT row = item.get_local_id(0); row < m; row += lsize) {
          IDXT tmp    = d_rowptr[row];
          IDXT rownnz = d_rowptr[row + 1] - tmp;
          IDXT idxg   = block * blocknnz + tmp;
          for (IDXT j = 0; j < rownnz; j++) {
            real v = c * d_J[idxg + j];
            if (d_colind[tmp + j] == row) v += 1.0;
            d_Mbcsr[idxg + j] = v;
          }
        }
      }
    });

    // ---- Newton iterations: fused matvec (M*ycur) + residual ----
    for (int it = 0; it < newton; it++) {
      q.parallel_for(sycl::nd_range<1>(bgws, blws), [=](sycl::nd_item<1> item) {
        const IDXT ngroups = item.get_group_range(0);
        const IDXT lsize   = item.get_local_range(0);
        for (IDXT block = item.get_group(0); block < nblocks;
             block += ngroups) {
          for (IDXT row = item.get_local_id(0); row < m; row += lsize) {
            IDXT tmp    = d_rowptr[row];
            IDXT rownnz = d_rowptr[row + 1] - tmp;
            IDXT idxg   = block * blocknnz + tmp;
            IDXT rowg   = block * m + row;
            IDXT colg   = block * m;
            real sum    = 0;
            for (IDXT j = 0; j < rownnz; j++)
              sum += d_Mbcsr[idxg + j] * d_ycur[colg + d_colind[tmp + j]];
            d_Ax[rowg]  = sum;
            d_res[rowg] = ngamma * sum + (rl1 * d_zn1[rowg] + d_ycor[rowg]);
          }
        }
      });
    }

    // ---- fused reduce + advance: five res reductions, the in-place state
    //      advance, and the renorm max norm (slot 5), all in one pass. ----
    auto init = sycl::property::reduction::initialize_to_identity{};
    auto rDot = sycl::reduction(d_red + 0, sycl::plus<real>(), init);
    auto rWl2 = sycl::reduction(d_red + 1, sycl::plus<real>(), init);
    auto rMx  = sycl::reduction(d_red + 2, sycl::maximum<real>(), init);
    auto rL1  = sycl::reduction(d_red + 3, sycl::plus<real>(), init);
    auto rMn  = sycl::reduction(d_red + 4, sycl::minimum<real>(), init);
    auto rSy  = sycl::reduction(d_red + 5, sycl::maximum<real>(), init);
    q.parallel_for(sycl::range<1>(n), rDot, rWl2, rMx, rL1, rMn, rSy,
                   [=](sycl::id<1> idx, auto &dot, auto &wl2, auto &mx,
                       auto &l1, auto &mn, auto &sy) {
                     size_t i = idx;
                     real rv  = d_res[i];
                     real p   = rv * d_ewt[i];
                     real a   = sycl::fabs(rv);
                     dot.combine(p);
                     wl2.combine(p * p);
                     mx.combine(a);
                     l1.combine(a);
                     mn.combine(rv);
                     real yn   = d_ycur[i] - damp * rv; // state advance
                     d_ycur[i] = yn;
                     sy.combine(sycl::fabs(yn)); // renorm norm
                   });

    // ---- renormalization by the device-resident max norm (slot 5) ----
    q.parallel_for(sycl::nd_range<1>(sgws, slws), [=](sycl::nd_item<1> item) {
      size_t i = item.get_global_id(0);
      if (i < n) d_ycur[i] = d_ycur[i] * (1.0 / d_red[5]);
    });
  }

  q.wait();
  long ns_total = ns_since(start);

  // Copy the final-step reduction scalars back once, after timing.
  real hred[6];
  q.memcpy(hred, d_red, 6 * sizeof(real)).wait();
  out->dot = hred[0]; out->wl2 = hred[1]; out->mx = hred[2];
  out->l1 = hred[3];  out->mn = hred[4];

  // Copy the final-step results back for verification.
  q.memcpy(out->ewt, d_ewt, n * sizeof(real));
  q.memcpy(out->ewtsv, d_ewtsv, n * sizeof(real));
  q.memcpy(out->tempv, d_tempv, n * sizeof(real));
  q.memcpy(out->ftemp, d_ftemp, n * sizeof(real));
  q.memcpy(out->Mdiag, d_Mdiag, n * sizeof(real));
  q.memcpy(out->Mupd, d_Mupd, n * sizeof(real));
  q.memcpy(out->res, d_res, n * sizeof(real));
  q.memcpy(out->Mbcsr, d_Mbcsr, nnz_total * sizeof(real));
  q.memcpy(out->Ax, d_Ax, n * sizeof(real));
  q.memcpy(out->ycur, d_ycur, n * sizeof(real));
  q.wait();

  sycl::free(d_rowptr, q); sycl::free(d_colind, q);
  sycl::free(d_ycur, q); sycl::free(d_Vabstol, q); sycl::free(d_cflag, q);
  sycl::free(d_mm, q); sycl::free(d_zn1, q); sycl::free(d_ycor, q);
  sycl::free(d_fpred, q); sycl::free(d_ypred, q); sycl::free(d_Min, q);
  sycl::free(d_J, q);
  sycl::free(d_ewt, q); sycl::free(d_ewtsv, q); sycl::free(d_tempv, q);
  sycl::free(d_save, q); sycl::free(d_ftemp, q); sycl::free(d_yform, q);
  sycl::free(d_Mdiag, q); sycl::free(d_ybuild, q); sycl::free(d_Mupd, q);
  sycl::free(d_res, q); sycl::free(d_Mbcsr, q); sycl::free(d_Ax, q);
  sycl::free(d_red, q);
  return ns_total;
}
