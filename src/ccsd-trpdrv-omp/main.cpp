#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "reference.h"

/* Do not allow the test to allocate more than MAX_MEM gigabytes. */
#ifndef MAX_MEM
#define MAX_MEM 4
#endif

#define MIN(x,y) (x<y ? x : y)
#define MAX(x,y) (x>y ? x : y)

long ccsd_trpdrv(
    double * __restrict f1n, double * __restrict f1t,
    double * __restrict f2n, double * __restrict f2t,
    double * __restrict f3n, double * __restrict f3t,
    double * __restrict f4n, double * __restrict f4t,
    double * __restrict eorb,
    int    * __restrict ncor_, int * __restrict nocc_, int * __restrict nvir_,
    double * __restrict emp4_, double * __restrict emp5_,
    int    * __restrict a_, int * __restrict i_, int * __restrict j_, int * __restrict k_, int * __restrict klo_,
    double * __restrict tij, double * __restrict tkj, double * __restrict tia, double * __restrict tka,
    double * __restrict xia, double * __restrict xka, double * __restrict jia, double * __restrict jka,
    double * __restrict kia, double * __restrict kka, double * __restrict jij, double * __restrict jkj,
    double * __restrict kij, double * __restrict kkj,
    double * __restrict dintc1, double * __restrict dintx1, double * __restrict t1v1,
    double * __restrict dintc2, double * __restrict dintx2, double * __restrict t1v2);

double * make_array(int n)
{
  double * a = (double*) malloc(n*sizeof(double));
  for (int i=0; i<n; i++) {
    a[i] = drand48();
  }
  return a;
}

int main(int argc, char* argv[])
{
  int ncor, nocc, nvir;
  int maxiter = 100;
  int nkpass = 1;

  if (argc<3) {
    printf("Usage: ./test_cbody nocc nvir [maxiter] [nkpass]\n");
    return argc;
  } else {
    ncor = 0;
    nocc = atoi(argv[1]);
    nvir = atoi(argv[2]);
    if (argc>3) {
      maxiter = atoi(argv[3]);
      /* if negative, treat as "infinite" */
      if (maxiter<0) maxiter = 1<<30;
    }
    if (argc>4) {
      nkpass = atoi(argv[4]);
    }
  }

  if (nocc<1 || nvir<1) {
    printf("Arguments must be non-negative!\n");
    return 1;
  }

  printf("Test driver for cbody with nocc=%d, nvir=%d, maxiter=%d, nkpass=%d\n", nocc, nvir, maxiter, nkpass);

  const int nbf = ncor + nocc + nvir;
  const int lnvv = nvir * nvir;
  const int lnov = nocc * nvir;
  const int kchunk = (nocc - 1)/nkpass + 1;

  const double memory = (nbf+8.0*lnvv+
      lnvv+kchunk*lnvv+lnov*nocc+kchunk*lnov+lnov*nocc+kchunk*lnov+lnvv+
      kchunk*lnvv+lnvv+kchunk*lnvv+lnov*nocc+kchunk*lnov+lnov*nocc+
      kchunk*lnov+lnov+nvir*kchunk+nvir*nocc+
      6.0*lnvv)*sizeof(double);
  printf("This test requires %f GB of memory.\n", 1.0e-9*memory);

  if (1.0e-9*memory > MAX_MEM) {
    printf("You need to increase MAX_MEM (%d)\n", MAX_MEM);
    printf("or set nkpass (%d) to a larger number.\n", nkpass);
    return MAX_MEM;
  }

  srand48(2);
  double * eorb = make_array(nbf);
  double * f1n = make_array(lnvv);
  double * f2n = make_array(lnvv);
  double * f3n = make_array(lnvv);
  double * f4n = make_array(lnvv);
  double * f1t = make_array(lnvv);
  double * f2t = make_array(lnvv);
  double * f3t = make_array(lnvv);
  double * f4t = make_array(lnvv);
  double * Tij = make_array(lnvv);
  double * Tkj = make_array(kchunk*lnvv);
  double * Tia = make_array(lnov*nocc);
  double * Tka = make_array(kchunk*lnov);
  double * Xia = make_array(lnov*nocc);
  double * Xka = make_array(kchunk*lnov);
  double * Jia = make_array(lnvv);
  double * Jka = make_array(kchunk*lnvv);
  double * Kia = make_array(lnvv);
  double * Kka = make_array(kchunk*lnvv);
  double * Jij = make_array(lnov*nocc);
  double * Jkj = make_array(kchunk*lnov);
  double * Kij = make_array(lnov*nocc);
  double * Kkj = make_array(kchunk*lnov);
  double * Dja = make_array(lnov);
  double * Djka = make_array(nvir*kchunk);
  double * Djia = make_array(nvir*nocc);
  double * dintc1 = make_array(lnvv);
  double * dintc2 = make_array(lnvv);
  double * dintx1 = make_array(lnvv);
  double * dintx2 = make_array(lnvv);
  double * t1v1 = make_array(lnvv);
  double * t1v2 = make_array(lnvv);

  int ntimers = MIN(maxiter,nocc*nocc*nocc*nocc);
  double * timers = (double*) calloc(ntimers,sizeof(double));

  double emp4=0.0, emp5=0.0;
  double emp4_r=0.0, emp5_r=0.0;

  int iter = 0;

  for (int klo=1; klo<=nocc; klo+=kchunk) {
    const int khi = MIN(nocc, klo+kchunk-1);
    int a=1;
    for (int j=1; j<=nocc; j++) {
      for (int i=1; i<=nocc; i++) {
        for (int k=klo; k<=MIN(khi,i); k++) {
          long time = ccsd_trpdrv(f1n, f1t, f2n, f2t, f3n, f3t, f4n, f4t, eorb,
              &ncor, &nocc, &nvir, &emp4, &emp5, &a, &i, &j, &k, &klo,
              Tij, Tkj, Tia, Tka, Xia, Xka, Jia, Jka, Kia, Kka, Jij, Jkj, Kij, Kkj,
              dintc1, dintx1, t1v1, dintc2, dintx2, t1v2);
          timers[iter] = time * 1e-9f;

          ccsd_trpdrv_ref(f1n, f1t, f2n, f2t, f3n, f3t, f4n, f4t, eorb,
              &ncor, &nocc, &nvir, &emp4_r, &emp5_r, &a, &i, &j, &k, &klo,
              Tij, Tkj, Tia, Tka, Xia, Xka, Jia, Jka, Kia, Kka, Jij, Jkj, Kij, Kkj,
              dintc1, dintx1, t1v1, dintc2, dintx2, t1v2);

          iter++;
          if (iter==maxiter) {
            printf("Stopping after %d iterations...\n", iter);
            goto maxed_out;
          }

          /* prevent NAN for large maxiter... */
          if (emp4 >  1000.0) emp4 -= 1000.0;
          if (emp4 < -1000.0) emp4 += 1000.0;
          if (emp5 >  1000.0) emp5 -= 1000.0;
          if (emp5 < -1000.0) emp5 += 1000.0;
          if (emp4_r >  1000.0) emp4_r -= 1000.0;
          if (emp4_r < -1000.0) emp4_r += 1000.0;
          if (emp5_r >  1000.0) emp5_r -= 1000.0;
          if (emp5_r < -1000.0) emp5_r += 1000.0;
        }
      }
    }
  }

maxed_out:
  double tsum =  0.0;
  double tmax = -1.0e10;
  double tmin =  1.0e10;
  for (int i=0; i<iter; i++) {
    tsum += timers[i];
    tmax  = MAX(tmax,timers[i]);
    tmin  = MIN(tmin,timers[i]);
  }
  double tavg = tsum / iter;
  printf("Kernel timing: min=%lf, max=%lf, avg=%lf\n", tmin, tmax, tavg);

  double dgemm_flops = ((8.0*nvir)*nvir)*(nvir+nocc);
  double dgemm_mops  = 8.0*(4.0*nvir*nvir + 2.0*nvir*nocc);

  /* The inner loop of tengy touches 86 f[1234][nt] elements and 8 other arrays...
   * We will just assume flops=mops even though flops>mops */
  double tengy_ops = ((1.0*nvir)*nvir)*(86+8);

  printf("OPS: dgemm_flops=%10.3e dgemm_mops=%10.3e tengy_ops=%10.3e\n",
         dgemm_flops, dgemm_mops, tengy_ops);
  printf("PERF: GF/s=%10.3e GB/s=%10.3e\n",
         1.0e-9*(dgemm_flops+tengy_ops)/tavg, 8.0e-9*(dgemm_mops+tengy_ops)/tavg);

  printf("These are meaningless but should not vary for a particular input:\n");
  printf("emp4=%f emp5=%f\n", emp4, emp5);

  bool ok = fabs(emp4_r - emp4) < 1e-6 && fabs(emp5_r - emp5) < 1e-6;
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  free(eorb);
  free(f1n );
  free(f2n );
  free(f3n );
  free(f4n );
  free(f1t );
  free(f2t );
  free(f3t );
  free(f4t );
  free(Tij );
  free(Tkj );
  free(Tia );
  free(Tka );
  free(Xia );
  free(Xka );
  free(Jia );
  free(Jka );
  free(Kia );
  free(Kka );
  free(Jij );
  free(Jkj );
  free(Kij );
  free(Kkj );
  free(Dja );
  free(Djka);
  free(Djia);
  free(dintc1);
  free(dintc2);
  free(dintx1);
  free(dintx2);
  free(t1v1  );
  free(t1v2  );

  return 0;
}
