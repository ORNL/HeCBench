#include <cstdio>
#include <cstdlib>
#include "sundials.h"

static REAL *halloc(size_t n) { return (REAL *)malloc(n * sizeof(REAL)); }

static void alloc_out(sun_out *o, size_t n, size_t nnz_total) {
  o->ewt   = halloc(n); o->ewtsv = halloc(n); o->tempv = halloc(n);
  o->ftemp = halloc(n); o->Mdiag = halloc(n); o->Mupd  = halloc(n);
  o->res   = halloc(n); o->Ax    = halloc(n); o->ycur  = halloc(n);
  o->Mbcsr = halloc(nnz_total);
}
static void free_out(sun_out *o) {
  free(o->ewt); free(o->ewtsv); free(o->tempv); free(o->ftemp);
  free(o->Mdiag); free(o->Mupd); free(o->res); free(o->Ax); free(o->ycur);
  free(o->Mbcsr);
}

int main(int argc, char *argv[]) {
  if (argc < 5 || argc > 6) {
    printf("Usage: %s <block dim m> <number of blocks> <nnz per row> "
           "<time steps> [newton iters]\n",
           argv[0]);
    return 1;
  }

  const IDXT m           = atoi(argv[1]);
  const IDXT nblocks     = atoi(argv[2]);
  const IDXT nnz_per_row = atoi(argv[3]);
  const int  steps       = atoi(argv[4]);
  const int  newton      = (argc == 6) ? atoi(argv[5]) : 3;
  const int  block_size  = 256;

  if (m <= 0 || nblocks <= 0 || nnz_per_row <= 0 || steps <= 0 || newton <= 0) {
    printf("Error: all arguments must be positive\n");
    return 1;
  }

  IDXT *rowptr = (IDXT *)malloc((m + 1) * sizeof(IDXT));
  IDXT *colind = (IDXT *)malloc((size_t)m * nnz_per_row * sizeof(IDXT));
  const IDXT blocknnz = init_pattern(m, nnz_per_row, rowptr, colind);

  const size_t n         = (size_t)m * nblocks;
  const size_t nnz_total = (size_t)nblocks * blocknnz;

  // Inputs
  REAL *Jvals   = halloc(nnz_total);
  REAL *ycur0   = halloc(n);
  REAL *Vabstol = halloc(n);
  REAL *cflag   = halloc(n);
  REAL *mm      = halloc(n);
  REAL *zn1     = halloc(n);
  REAL *ycor    = halloc(n);
  REAL *fpred   = halloc(n);
  REAL *ypred   = halloc(n);
  REAL *Min     = halloc(n);

  init_values(Jvals, nnz_total, 11);
  init_data(ycur0, n, 1, -2.0, 2.0);
  init_data(Vabstol, n, 2, 1e-9, 1e-7);
  init_data(cflag, n, 3, -2.0, 2.0);
  init_data(mm, n, 4, 0.0, 1.0);
  init_data(zn1, n, 5, -1.0, 1.0);
  init_data(ycor, n, 6, -1.0, 1.0);
  init_data(fpred, n, 7, 0.5, 1.5);
  init_data(ypred, n, 8, -1.0, 1.0);
  init_data(Min, n, 9, 0.5, 1.5);

  sun_in in = {};
  in.m = m; in.nblocks = nblocks; in.nnz_per_row = nnz_per_row;
  in.blocknnz = blocknnz;
  in.rowptr = rowptr; in.colind = colind; in.Jvals = Jvals;
  in.ycur0 = ycur0; in.Vabstol = Vabstol; in.cflag = cflag; in.mm = mm;
  in.zn1 = zn1; in.ycor = ycor; in.fpred = fpred; in.ypred = ypred;
  in.Min = Min;
  in.reltol = 1e-2; in.Sabstol = 1e-3; in.rl1 = 0.5; in.ngamma = -0.1;
  in.h = 1e-3; in.r = 0.75; in.fract = 0.1; in.uround = 1e-13;
  in.gamma = 0.1; in.damp = 0.1;

  printf("SUNDIALS implicit-integration miniapp\n");
  printf("Batch: %d systems of size %d  (vector length n = %zu)\n", nblocks, m,
         n);
  printf("Block-CSR: %d nonzeros/block (%d per row), %zu total nonzeros\n",
         blocknnz, blocknnz / m, nnz_total);
  printf("Time steps: %d, Newton iterations/step: %d\n\n", steps, newton);

  sun_out dev, ref;
  alloc_out(&dev, n, nnz_total);
  alloc_out(&ref, n, nnz_total);

  // Warm up the device/context (results discarded).
  sundials_miniapp(2, newton, block_size, &in, &dev);

  long ns_total = sundials_miniapp(steps, newton, block_size, &in, &dev);
  reference(steps, newton, &in, &ref);

  // ---- timing (whole step loop) ----
  printf("Total step-loop time: %.4f ms  (%.4f ms/step)\n\n", ns_total * 1e-6,
         ns_total * 1e-6 / steps);

  // ---- verification (final-step results, device vs host reference) ----
  const double tol = 1e-6;
  bool pass = true;

  struct { const char *name; double err; } arr[] = {
      {"ewtSetSS",        check(dev.ewt, ref.ewt, n)},
      {"ewtSetSV",        check(dev.ewtsv, ref.ewtsv, n)},
      {"checkConstraints",check(dev.tempv, ref.tempv, n)},
      {"diagSetupFormY",  check(dev.ftemp, ref.ftemp, n)},
      {"diagSetupBuildM", check(dev.Mdiag, ref.Mdiag, n)},
      {"diagSolveUpdateM",check(dev.Mupd, ref.Mupd, n)},
      {"nlsResid",        check(dev.res, ref.res, n)},
      {"scaleAddI",       check(dev.Mbcsr, ref.Mbcsr, nnz_total)},
      {"matvec",          check(dev.Ax, ref.Ax, n)},
      {"state (ycur)",    check(dev.ycur, ref.ycur, n)},
  };
  struct { const char *name; double err; } scl[] = {
      {"dotProd",       rel_err(dev.dot, ref.dot)},
      {"wL2NormSquare", rel_err(dev.wl2, ref.wl2)},
      {"maxNorm",       rel_err(dev.mx, ref.mx)},
      {"L1Norm",        rel_err(dev.l1, ref.l1)},
      {"findMin",       rel_err(dev.mn, ref.mn)},
  };

  printf("%-22s %14s %8s\n", "kernel", "rel err", "status");
  for (auto &a : arr) {
    bool p = a.err <= tol;
    pass   = pass && p;
    printf("%-22s %14.3e %8s\n", a.name, a.err, p ? "PASS" : "FAIL");
  }
  for (auto &sc : scl) {
    bool p = sc.err <= tol;
    pass   = pass && p;
    printf("%-22s %14.3e %8s\n", sc.name, sc.err, p ? "PASS" : "FAIL");
  }

  printf("\n%s\n", pass ? "PASS" : "FAIL");

  free_out(&dev); free_out(&ref);
  free(rowptr); free(colind);
  free(Jvals); free(ycur0); free(Vabstol); free(cflag); free(mm);
  free(zn1); free(ycor); free(fpred); free(ypred); free(Min);
  return pass ? 0 : 1;
}
