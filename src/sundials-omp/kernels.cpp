#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include "sundials.h"

/*
 * =============================================================================
 * OpenMP target-offload port of the SUNDIALS implicit-integration miniapp.
 * =============================================================================
 */

using clk = std::chrono::steady_clock;
static inline long ns_since(clk::time_point t0) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(clk::now() - t0)
      .count();
}

long sundials_miniapp(int steps, int newton, int /*bs*/, const sun_in *in,
                      sun_out *out) {
  const IDXT m = in->m, nblocks = in->nblocks, blocknnz = in->blocknnz;
  const size_t n         = (size_t)m * nblocks;
  const size_t nnz_total = (size_t)nblocks * blocknnz;

  const real reltol = in->reltol, Sabstol = in->Sabstol, rl1 = in->rl1,
             ngamma = in->ngamma, h = in->h, r = in->r, fract = in->fract,
             uround = in->uround, c = -in->gamma, damp = in->damp;

  // Inputs (resident on the device for the whole loop).
  const IDXT *rowptr = in->rowptr, *colind = in->colind;
  const REAL *Jvals = in->Jvals, *Vabstol = in->Vabstol, *cflag = in->cflag,
             *mm = in->mm, *zn1 = in->zn1, *ycor = in->ycor, *fpred = in->fpred,
             *ypred = in->ypred, *Min = in->Min;

  // Outputs (caller-allocated host buffers); copied back at region exit.
  real *ewt = out->ewt, *ewtsv = out->ewtsv, *tempv = out->tempv,
       *ftemp = out->ftemp, *Mdiag = out->Mdiag, *Mupd = out->Mupd,
       *res = out->res, *Mbcsr = out->Mbcsr, *Ax = out->Ax, *ycur = out->ycur;

  // Device-only scratch (written but not verified on the host).
  real *save   = (real *)malloc(n * sizeof(real));
  real *yform  = (real *)malloc(n * sizeof(real));
  real *ybuild = (real *)malloc(n * sizeof(real));

  // ycur is the evolving state; seed it from the initial condition.
  for (size_t i = 0; i < n; i++) ycur[i] = in->ycur0[i];

  double dot = 0, wl2 = 0, mx = 0, l1 = 0, mn = 0;
  long ns = 0;

  #pragma omp target data                                                      \
      map(to: rowptr[0:m + 1], colind[0:blocknnz], Jvals[0:nnz_total],         \
              Vabstol[0:n], cflag[0:n], mm[0:n], zn1[0:n], ycor[0:n],          \
              fpred[0:n], ypred[0:n], Min[0:n])                                \
      map(tofrom: ycur[0:n])                                                   \
      map(from: ewt[0:n], ewtsv[0:n], tempv[0:n], ftemp[0:n], Mdiag[0:n],      \
                Mupd[0:n], res[0:n], Mbcsr[0:nnz_total], Ax[0:n])              \
      map(alloc: save[0:n], yform[0:n], ybuild[0:n])
  {
    clk::time_point start = clk::now();

    for (int s = 0; s < steps; s++) {
      // ---- fused setup: ewtSS/SV + checkConstraints + CVDiag form/build/update
      #pragma omp target teams distribute parallel for
      for (size_t i = 0; i < n; i++) {
        real yc  = ycur[i];
        real w   = 1.0 / (reltol * fabs(yc) + Sabstol);
        ewt[i]   = w;
        ewtsv[i] = 1.0 / (reltol * fabs(yc) + Vabstol[i]);

        real tmp = (fabs(cflag[i]) >= 1.5) ? 1.0 : 0.0;
        tmp      = tmp * cflag[i] / w;
        save[i]  = -0.1 * tmp;
        tempv[i] = (yc - 0.1 * tmp) * mm[i];

        real ft  = h * fpred[i] - zn1[i];
        ftemp[i] = ft;
        yform[i] = r * ft + ypred[i];

        real mval    = fract * ft - h * (Min[i] - fpred[i]);
        real yv      = ft * w;
        bool test    = (fabs(yv) >= uround);
        real bit     = test ? 1.0 : 0.0;
        real bitcomp = test ? 0.0 : -1.0;
        yv           = fract * ft * bit - bitcomp;
        real md      = mval / yv * bit - bitcomp;
        Mdiag[i]     = md;
        ybuild[i]    = yv;

        Mupd[i] = r * (1.0 / md - 1.0) + 1.0;
      }

      // ---- scaleAddI: form Newton matrix M = c*J + I (BCSR) ----
      #pragma omp target teams distribute parallel for collapse(2)
      for (IDXT block = 0; block < nblocks; block++) {
        for (IDXT row = 0; row < m; row++) {
          IDXT tmp    = rowptr[row];
          IDXT rownnz = rowptr[row + 1] - tmp;
          IDXT idxg   = block * blocknnz + tmp;
          for (IDXT j = 0; j < rownnz; j++) {
            REAL v = c * Jvals[idxg + j];
            if (colind[tmp + j] == row) v += 1.0;
            Mbcsr[idxg + j] = v;
          }
        }
      }

      // ---- Newton iterations: fused matvec (M*ycur) + residual ----
      for (int it = 0; it < newton; it++) {
        #pragma omp target teams distribute parallel for collapse(2)
        for (IDXT block = 0; block < nblocks; block++) {
          for (IDXT row = 0; row < m; row++) {
            IDXT tmp    = rowptr[row];
            IDXT rownnz = rowptr[row + 1] - tmp;
            IDXT idxg   = block * blocknnz + tmp;
            IDXT rowg   = block * m + row;
            IDXT colg   = block * m;
            REAL sum    = 0;
            for (IDXT j = 0; j < rownnz; j++)
              sum += Mbcsr[idxg + j] * ycur[colg + colind[tmp + j]];
            Ax[rowg]  = sum;
            res[rowg] = ngamma * sum + (rl1 * zn1[rowg] + ycor[rowg]);
          }
        }
      }

      // ---- fused reduce + advance: 5 res reductions + state advance + renorm.
      double sdot = 0, swl2 = 0, sl1 = 0;
      double smx = -DBL_MAX, smn = DBL_MAX, symx = -DBL_MAX;
      #pragma omp target teams distribute parallel for                         \
          reduction(+: sdot, swl2, sl1) reduction(max: smx, symx)              \
          reduction(min: smn)
      for (size_t i = 0; i < n; i++) {
        real rv = res[i], p = rv * ewt[i], a = fabs(rv);
        sdot += p; swl2 += p * p; sl1 += a;
        smx = fmax(a, smx); smn = fmin(rv, smn);
        real yn = ycur[i] - damp * rv;
        ycur[i] = yn;
        symx = fmax(fabs(yn), symx);
      }

      // Renormalize the advanced state (power-iteration style).
      real inv = 1.0 / symx;
      #pragma omp target teams distribute parallel for
      for (size_t i = 0; i < n; i++) ycur[i] = ycur[i] * inv;

      dot = sdot; wl2 = swl2; mx = smx; l1 = sl1; mn = smn;
    }

    ns = ns_since(start);
  }

  out->dot = dot; out->wl2 = wl2; out->mx = mx; out->l1 = l1; out->mn = mn;
  free(save); free(yform); free(ybuild);
  return ns;
}
