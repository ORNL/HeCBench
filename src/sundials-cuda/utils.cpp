#include <cmath>
#include <cstdlib>
#include "sundials.h"

/*
 * =============================================================================
 * Shared init / verification helpers
 * =============================================================================
 */

// Build a shared m x m sparsity pattern: each row has exactly nnz_per_row
// nonzeros including the diagonal. Column indices per row are sorted.
IDXT init_pattern(IDXT m, IDXT nnz_per_row, IDXT *rowptr, IDXT *colind) {
  if (nnz_per_row > m) nnz_per_row = m;

  srand(1234);
  IDXT idx    = 0;
  char *used  = (char *)malloc(m * sizeof(char));

  for (IDXT row = 0; row < m; row++) {
    rowptr[row] = idx;
    for (IDXT j = 0; j < m; j++) used[j] = 0;

    used[row]  = 1; // diagonal always present (required by scaleAddI)
    IDXT count = 1;
    while (count < nnz_per_row) {
      IDXT col = rand() % m;
      if (!used[col]) { used[col] = 1; count++; }
    }
    for (IDXT j = 0; j < m; j++)
      if (used[j]) colind[idx++] = j;
  }
  rowptr[m] = idx;

  free(used);
  return idx; // blocknnz
}

void init_values(REAL *a, size_t n, unsigned seed) {
  srand48((long)seed);
  for (size_t i = 0; i < n; i++) a[i] = 0.5 + (REAL)drand48();
}

void init_data(real *a, size_t n, unsigned seed, real lo, real hi) {
  srand48((long)seed);
  for (size_t i = 0; i < n; i++) a[i] = lo + (hi - lo) * (real)drand48();
}

double check(const REAL *a, const REAL *b, size_t n) {
  double diff = 0, sum = 0;
  for (size_t i = 0; i < n; i++) {
    diff += std::fabs((double)a[i] - (double)b[i]);
    sum  += std::fabs((double)b[i]);
  }
  return (sum > 0) ? diff / sum : diff;
}

double rel_err(real a, real b) {
  double d = std::fabs((double)a - (double)b);
  double s = std::fabs((double)b);
  return (s > 0) ? d / s : d;
}

/*
 * =============================================================================
 * Host reference: identical serial sequence to sundials_miniapp (kernels.cu)
 * =============================================================================
 */
void reference(int steps, int newton, const sun_in *in, sun_out *out) {
  const IDXT m = in->m, nblocks = in->nblocks, blocknnz = in->blocknnz;
  const size_t n         = (size_t)m * nblocks;
  const size_t nnz_total = (size_t)nblocks * blocknnz;

  const real reltol = in->reltol, Sabstol = in->Sabstol, rl1 = in->rl1,
             ngamma = in->ngamma, h = in->h, r = in->r, fract = in->fract,
             uround = in->uround, c = -in->gamma, damp = in->damp;

  // Working state (evolves across steps).
  real *ycur = (real *)malloc(n * sizeof(real));
  for (size_t i = 0; i < n; i++) ycur[i] = in->ycur0[i];

  // Per-step scratch reused every step (out->* hold the final snapshot).
  real *yform  = (real *)malloc(n * sizeof(real));
  real *ybuild = (real *)malloc(n * sizeof(real));
  real *save   = (real *)malloc(n * sizeof(real));

  real *ewt = out->ewt, *ewtsv = out->ewtsv, *tempv = out->tempv,
       *ftemp = out->ftemp, *Mdiag = out->Mdiag, *Mupd = out->Mupd,
       *res = out->res, *Mbcsr = out->Mbcsr, *Ax = out->Ax;

  for (int s = 0; s < steps; s++) {
    // streaming: error-weight setup + CVDiag preconditioner setup
    for (size_t i = 0; i < n; i++) {
      ewt[i]   = 1.0 / (reltol * std::fabs(ycur[i]) + Sabstol);
      ewtsv[i] = 1.0 / (reltol * std::fabs(ycur[i]) + in->Vabstol[i]);
    }
    for (size_t i = 0; i < n; i++) {
      real tmp = (std::fabs(in->cflag[i]) >= 1.5) ? 1.0 : 0.0;
      tmp      = tmp * in->cflag[i];
      tmp      = tmp / ewt[i];
      save[i]  = -0.1 * tmp;
      tmp      = ycur[i] - 0.1 * tmp;
      tempv[i] = tmp * in->mm[i];
    }
    for (size_t i = 0; i < n; i++) {
      ftemp[i] = h * in->fpred[i] - in->zn1[i];
      yform[i] = r * ftemp[i] + in->ypred[i];
    }
    for (size_t i = 0; i < n; i++) {
      real mval    = fract * ftemp[i] - h * (in->Min[i] - in->fpred[i]);
      real yv      = ftemp[i] * ewt[i];
      bool test    = (std::fabs(yv) >= uround);
      real bit     = test ? 1.0 : 0.0;
      real bitcomp = test ? 0.0 : -1.0;
      yv           = fract * ftemp[i] * bit - bitcomp;
      Mdiag[i]     = mval / yv * bit - bitcomp;
      ybuild[i]    = yv;
    }
    for (size_t i = 0; i < n; i++)
      Mupd[i] = r * (1.0 / Mdiag[i] - 1.0) + 1.0;

    // scaleAddI: form Newton matrix M = c*J + I (BCSR)
    for (size_t i = 0; i < nnz_total; i++) Mbcsr[i] = in->Jvals[i];
    for (IDXT block = 0; block < nblocks; block++) {
      for (IDXT row = 0; row < m; row++) {
        IDXT tmp    = in->rowptr[row];
        IDXT rownnz = in->rowptr[row + 1] - tmp;
        IDXT idxg   = block * blocknnz + tmp;
        for (IDXT j = 0; j < rownnz; j++) {
          if (in->colind[tmp + j] == row) Mbcsr[idxg + j] = c * Mbcsr[idxg + j] + 1.0;
          else                            Mbcsr[idxg + j] = c * Mbcsr[idxg + j];
        }
      }
    }

    // Newton iterations: matvec (M*ycur) + fused residual
    for (int it = 0; it < newton; it++) {
      for (IDXT block = 0; block < nblocks; block++) {
        for (IDXT row = 0; row < m; row++) {
          IDXT tmp    = in->rowptr[row];
          IDXT rownnz = in->rowptr[row + 1] - tmp;
          IDXT idxg   = block * blocknnz + tmp;
          IDXT colg   = block * m;
          real sum    = 0;
          for (IDXT j = 0; j < rownnz; j++)
            sum += Mbcsr[idxg + j] * ycur[colg + in->colind[tmp + j]];
          Ax[block * m + row] = sum;
        }
      }
      for (size_t i = 0; i < n; i++)
        res[i] = ngamma * Ax[i] + (rl1 * in->zn1[i] + in->ycor[i]);
    }

    // reductions: WRMS-norm error control + diagnostics (on res)
    double dot = 0, wl2 = 0, l1 = 0, mx = 0, mn = res[0];
    for (size_t i = 0; i < n; i++) {
      double p = (double)res[i] * (double)ewt[i];
      dot += p;
      wl2 += p * p;
      double a = std::fabs((double)res[i]);
      l1 += a;
      if (a > mx) mx = a;
      if ((double)res[i] < mn) mn = (double)res[i];
    }
    out->dot = dot; out->wl2 = wl2; out->l1 = l1; out->mx = mx; out->mn = mn;

    // streaming: advance state, then renormalize (power-iteration)
    for (size_t i = 0; i < n; i++) ycur[i] = ycur[i] - damp * res[i];
    double nrm = 0;
    for (size_t i = 0; i < n; i++) {
      double a = std::fabs((double)ycur[i]);
      if (a > nrm) nrm = a;
    }
    for (size_t i = 0; i < n; i++) ycur[i] = ycur[i] * (1.0 / nrm);
  }

  for (size_t i = 0; i < n; i++) out->ycur[i] = ycur[i];

  free(ycur); free(yform); free(ybuild); free(save);
}
