/*****************************************************************************
 Self-contained replacement for the subset of the GNU Scientific Library used
 by this benchmark, added so that the benchmark no longer links against the
 GPL-licensed GSL (see HeCBench issue #319).

 Two facilities are provided, both under the BSD 3-Clause license of this
 benchmark:

   1. The MT19937 generator of Matsumoto and Nishimura, seeded and tempered
      exactly as GSL's gsl_rng_mt19937 does, so that the random stream (and
      therefore the stochastic search performed by this benchmark) is
      unchanged.

   2. Adaptive Gauss-Kronrod quadrature over a semi-infinite interval,
      following the QAGS/QAGI algorithm of QUADPACK (Piessens, de
      Doncker-Kapenga, Ueberhuber and Kahaner; public domain), which is the
      algorithm GSL's gsl_integration_qagiu implements.
 *****************************************************************************/

#ifndef GSL_COMPAT_H
#define GSL_COMPAT_H

#include <cmath>
#include <cstddef>
#include <cstdlib>

#define GSL_DBL_EPSILON 2.2204460492503131e-16
#define GSL_DBL_MIN 2.2250738585072014e-308
#define GSL_DBL_MAX 1.7976931348623157e+308

#define GSL_MAX_DBL(a, b) ((a) > (b) ? (a) : (b))

#define GSL_SUCCESS 0
#define GSL_EINVAL 4
#define GSL_EMAXITER 11
#define GSL_EBADTOL 13
#define GSL_EROUND 18
#define GSL_ESING 21
#define GSL_EDIVERGE 22

//--------------------------------------------------------------------
// Random number generation: MT19937
//--------------------------------------------------------------------

typedef struct {
  const char *name;
  unsigned long int max;
  unsigned long int min;
} gsl_rng_type;

#define GSL_COMPAT_MT_N 624
#define GSL_COMPAT_MT_M 397

typedef struct {
  const gsl_rng_type *type;
  unsigned long int mt[GSL_COMPAT_MT_N];
  int mti;
} gsl_rng;

inline const gsl_rng_type gsl_rng_mt19937_type = {"mt19937", 0xffffffffUL, 0UL};
inline const gsl_rng_type *gsl_rng_mt19937 = &gsl_rng_mt19937_type;
inline const gsl_rng_type *gsl_rng_default = &gsl_rng_mt19937_type;
inline unsigned long int gsl_rng_default_seed = 0;

inline void gsl_rng_set(gsl_rng *r, unsigned long int s) {
  // 4357 is the seed GSL substitutes for zero
  if (s == 0) s = 4357;

  r->mt[0] = s & 0xffffffffUL;

  int i;
  for (i = 1; i < GSL_COMPAT_MT_N; i++) {
    r->mt[i] = (1812433253UL * (r->mt[i - 1] ^ (r->mt[i - 1] >> 30)) +
                (unsigned long int)i) & 0xffffffffUL;
  }
  r->mti = i;
}

inline unsigned long int gsl_rng_get(gsl_rng *r) {
  const unsigned long int UPPER_MASK = 0x80000000UL;
  const unsigned long int LOWER_MASK = 0x7fffffffUL;
  unsigned long int *const mt = r->mt;

  if (r->mti >= GSL_COMPAT_MT_N) {
    int kk;
    unsigned long int y;

    for (kk = 0; kk < GSL_COMPAT_MT_N - GSL_COMPAT_MT_M; kk++) {
      y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
      mt[kk] = mt[kk + GSL_COMPAT_MT_M] ^ (y >> 1) ^
               ((y & 0x1UL) ? 0x9908b0dfUL : 0UL);
    }
    for (; kk < GSL_COMPAT_MT_N - 1; kk++) {
      y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
      mt[kk] = mt[kk + (GSL_COMPAT_MT_M - GSL_COMPAT_MT_N)] ^ (y >> 1) ^
               ((y & 0x1UL) ? 0x9908b0dfUL : 0UL);
    }
    y = (mt[GSL_COMPAT_MT_N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
    mt[GSL_COMPAT_MT_N - 1] = mt[GSL_COMPAT_MT_M - 1] ^ (y >> 1) ^
                              ((y & 0x1UL) ? 0x9908b0dfUL : 0UL);

    r->mti = 0;
  }

  // Tempering. The state is held masked to 32 bits, and both shift-and-mask
  // steps use 32-bit masks, so k stays within 32 bits throughout.
  unsigned long int k = mt[r->mti];
  k ^= (k >> 11);
  k ^= (k << 7) & 0x9d2c5680UL;
  k ^= (k << 15) & 0xefc60000UL;
  k ^= (k >> 18);

  r->mti++;

  return k;
}

inline double gsl_rng_uniform(gsl_rng *r) {
  return gsl_rng_get(r) / 4294967296.0;
}

inline gsl_rng *gsl_rng_alloc(const gsl_rng_type *T) {
  gsl_rng *r = (gsl_rng *)malloc(sizeof(gsl_rng));
  if (r == NULL) return NULL;
  r->type = T;
  gsl_rng_set(r, gsl_rng_default_seed);
  return r;
}

inline void gsl_rng_free(gsl_rng *r) { free(r); }

// Honours GSL_RNG_SEED so that the command lines shipped with the benchmark
// keep working. GSL_RNG_TYPE is ignored: only the default generator exists.
inline const gsl_rng_type *gsl_rng_env_setup(void) {
  const char *seed = getenv("GSL_RNG_SEED");
  if (seed != NULL) {
    gsl_rng_default_seed = strtoul(seed, NULL, 0);
  }
  return gsl_rng_default;
}

inline double gsl_ran_flat(gsl_rng *r, const double a, const double b) {
  double u = gsl_rng_uniform(r);
  return a * (1 - u) + b * u;
}

//--------------------------------------------------------------------
// Numerical integration
//--------------------------------------------------------------------

typedef struct {
  double (*function)(double x, void *params);
  void *params;
} gsl_function;

#define GSL_FN_EVAL(F, x) (*((F)->function))((x), (F)->params)

typedef struct {
  size_t limit;
  size_t size;
  size_t nrmax;
  size_t i;
  size_t maximum_level;
  double *alist;
  double *blist;
  double *rlist;
  double *elist;
  size_t *order;
  size_t *level;
} gsl_integration_workspace;

inline gsl_integration_workspace *
gsl_integration_workspace_alloc(const size_t n) {
  if (n == 0) return NULL;

  gsl_integration_workspace *w =
      (gsl_integration_workspace *)malloc(sizeof(gsl_integration_workspace));
  if (w == NULL) return NULL;

  w->alist = (double *)malloc(n * sizeof(double));
  w->blist = (double *)malloc(n * sizeof(double));
  w->rlist = (double *)malloc(n * sizeof(double));
  w->elist = (double *)malloc(n * sizeof(double));
  w->order = (size_t *)malloc(n * sizeof(size_t));
  w->level = (size_t *)malloc(n * sizeof(size_t));

  w->limit = n;
  w->size = 0;
  w->nrmax = 0;
  w->i = 0;
  w->maximum_level = 0;

  return w;
}

inline void gsl_integration_workspace_free(gsl_integration_workspace *w) {
  if (w == NULL) return;
  free(w->level);
  free(w->order);
  free(w->elist);
  free(w->rlist);
  free(w->blist);
  free(w->alist);
  free(w);
}

//---------------- 15-point Gauss-Kronrod rule -----------------------

inline double gsl_compat_rescale_error(double err, const double result_abs,
                                       const double result_asc) {
  err = fabs(err);

  if (result_asc != 0 && err != 0) {
    double scale = pow((200 * err / result_asc), 1.5);
    if (scale < 1)
      err = result_asc * scale;
    else
      err = result_asc;
  }

  if (result_abs > GSL_DBL_MIN / (50 * GSL_DBL_EPSILON)) {
    double min_err = 50 * GSL_DBL_EPSILON * result_abs;
    if (min_err > err) err = min_err;
  }

  return err;
}

inline void gsl_integration_qk15(const gsl_function *f, double a, double b,
                                 double *result, double *abserr, double *resabs,
                                 double *resasc) {
  // Abscissae of the 15-point Kronrod rule
  static const double xgk[8] = {
      0.991455371120812639206854697526329,
      0.949107912342758524526189684047851,
      0.864864423359769072789712788640926,
      0.741531185599394439863864773280788,
      0.586087235467691130294144838258730,
      0.405845151377397166906606412076961,
      0.207784955007898467600689403773245,
      0.000000000000000000000000000000000};

  // Weights of the 15-point Kronrod rule
  static const double wgk[8] = {
      0.022935322010529224963732008058970,
      0.063092092629978553290700663189204,
      0.104790010322250183839876322541518,
      0.140653259715525918745189590510238,
      0.169004726639267902826583426598550,
      0.190350578064785409913256402421014,
      0.204432940075298892414161999234649,
      0.209482141084727828012999174891714};

  // Weights of the embedded 7-point Gauss rule
  static const double wg[4] = {0.129484966168869693270611432679082,
                               0.279705391489276667901467771423780,
                               0.381830050505118944950369775488975,
                               0.417959183673469387755102040816327};

  const int n = 8;
  double fv1[8], fv2[8];

  const double center = 0.5 * (a + b);
  const double half_length = 0.5 * (b - a);
  const double abs_half_length = fabs(half_length);
  const double f_center = GSL_FN_EVAL(f, center);

  double result_gauss = f_center * wg[n / 2 - 1];
  double result_kronrod = f_center * wgk[n - 1];
  double result_abs = fabs(result_kronrod);
  double result_asc = 0;

  int j;

  for (j = 0; j < (n - 1) / 2; j++) {
    const int jtw = j * 2 + 1;
    const double abscissa = half_length * xgk[jtw];
    const double fval1 = GSL_FN_EVAL(f, center - abscissa);
    const double fval2 = GSL_FN_EVAL(f, center + abscissa);
    const double fsum = fval1 + fval2;
    fv1[jtw] = fval1;
    fv2[jtw] = fval2;
    result_gauss += wg[j] * fsum;
    result_kronrod += wgk[jtw] * fsum;
    result_abs += wgk[jtw] * (fabs(fval1) + fabs(fval2));
  }

  for (j = 0; j < n / 2; j++) {
    const int jtwm1 = j * 2;
    const double abscissa = half_length * xgk[jtwm1];
    const double fval1 = GSL_FN_EVAL(f, center - abscissa);
    const double fval2 = GSL_FN_EVAL(f, center + abscissa);
    fv1[jtwm1] = fval1;
    fv2[jtwm1] = fval2;
    result_kronrod += wgk[jtwm1] * (fval1 + fval2);
    result_abs += wgk[jtwm1] * (fabs(fval1) + fabs(fval2));
  }

  const double mean = result_kronrod * 0.5;

  result_asc = wgk[n - 1] * fabs(f_center - mean);
  for (j = 0; j < n - 1; j++) {
    result_asc += wgk[j] * (fabs(fv1[j] - mean) + fabs(fv2[j] - mean));
  }

  const double err = (result_kronrod - result_gauss) * half_length;

  result_kronrod *= half_length;
  result_abs *= abs_half_length;
  result_asc *= abs_half_length;

  *result = result_kronrod;
  *resabs = result_abs;
  *resasc = result_asc;
  *abserr = gsl_compat_rescale_error(err, result_abs, result_asc);
}

//---------------- Subinterval bookkeeping ---------------------------

inline void gsl_compat_initialise(gsl_integration_workspace *w, double a,
                                  double b) {
  w->size = 0;
  w->nrmax = 0;
  w->i = 0;
  w->alist[0] = a;
  w->blist[0] = b;
  w->rlist[0] = 0.0;
  w->elist[0] = 0.0;
  w->order[0] = 0;
  w->level[0] = 0;
  w->maximum_level = 0;
}

inline void gsl_compat_set_initial_result(gsl_integration_workspace *w,
                                          double result, double error) {
  w->size = 1;
  w->rlist[0] = result;
  w->elist[0] = error;
}

// Maintains the list of subintervals sorted by descending error estimate.
inline void gsl_compat_qpsrt(gsl_integration_workspace *w) {
  const size_t last = w->size - 1;
  const size_t limit = w->limit;
  double *elist = w->elist;
  size_t *order = w->order;

  size_t i_nrmax = w->nrmax;
  size_t i_maxerr = order[i_nrmax];

  if (last < 2) {
    order[0] = 0;
    order[1] = 1;
    w->i = i_maxerr;
    return;
  }

  const double errmax = elist[i_maxerr];

  while (i_nrmax > 0 && errmax > elist[order[i_nrmax - 1]]) {
    order[i_nrmax] = order[i_nrmax - 1];
    i_nrmax--;
  }

  int top;
  if (last < (limit / 2 + 2))
    top = (int)last;
  else
    top = (int)(limit - last + 1);

  int i = (int)i_nrmax + 1;

  while (i < top && errmax < elist[order[i]]) {
    order[i - 1] = order[i];
    i++;
  }

  order[i - 1] = i_maxerr;

  const double errmin = elist[last];

  int k = top - 1;

  while (k > i - 2 && errmin >= elist[order[k]]) {
    order[k + 1] = order[k];
    k--;
  }

  order[k + 1] = last;

  i_maxerr = order[i_nrmax];

  w->i = i_maxerr;
  w->nrmax = i_nrmax;
}

inline void gsl_compat_retrieve(const gsl_integration_workspace *w, double *a,
                                double *b, double *r, double *e) {
  const size_t i = w->i;
  *a = w->alist[i];
  *b = w->blist[i];
  *r = w->rlist[i];
  *e = w->elist[i];
}

inline void gsl_compat_update(gsl_integration_workspace *w, double a1, double b1,
                              double area1, double error1, double a2, double b2,
                              double area2, double error2) {
  const size_t i_max = w->i;
  const size_t i_new = w->size;
  const size_t new_level = w->level[i_max] + 1;

  if (error2 > error1) {
    w->alist[i_max] = a2; // blist[i_max] already holds b2
    w->rlist[i_max] = area2;
    w->elist[i_max] = error2;
    w->level[i_max] = new_level;

    w->alist[i_new] = a1;
    w->blist[i_new] = b1;
    w->rlist[i_new] = area1;
    w->elist[i_new] = error1;
    w->level[i_new] = new_level;
  } else {
    w->blist[i_max] = b1; // alist[i_max] already holds a1
    w->rlist[i_max] = area1;
    w->elist[i_max] = error1;
    w->level[i_max] = new_level;

    w->alist[i_new] = a2;
    w->blist[i_new] = b2;
    w->rlist[i_new] = area2;
    w->elist[i_new] = error2;
    w->level[i_new] = new_level;
  }

  w->size++;

  if (new_level > w->maximum_level) w->maximum_level = new_level;

  gsl_compat_qpsrt(w);
}

inline double gsl_compat_sum_results(const gsl_integration_workspace *w) {
  double result_sum = 0;
  for (size_t k = 0; k < w->size; k++) result_sum += w->rlist[k];
  return result_sum;
}

inline int gsl_compat_subinterval_too_small(double a1, double a2, double b2) {
  const double tmp =
      (1 + 100 * GSL_DBL_EPSILON) * (fabs(a2) + 1000 * GSL_DBL_MIN);
  return fabs(a1) <= tmp && fabs(b2) <= tmp;
}

inline void gsl_compat_reset_nrmax(gsl_integration_workspace *w) {
  w->nrmax = 0;
  w->i = w->order[0];
}

inline int gsl_compat_increase_nrmax(gsl_integration_workspace *w) {
  const size_t id = w->nrmax;
  const size_t last = w->size - 1;

  size_t jupbnd;
  if (last > (1 + w->limit / 2))
    jupbnd = w->limit + 1 - last;
  else
    jupbnd = last;

  for (size_t k = id; k <= jupbnd; k++) {
    const size_t i_max = w->order[w->nrmax];
    w->i = i_max;
    if (w->level[i_max] < w->maximum_level) return 1;
    w->nrmax++;
  }
  return 0;
}

inline int gsl_compat_large_interval(gsl_integration_workspace *w) {
  return w->level[w->i] < w->maximum_level;
}

//---------------- Wynn epsilon extrapolation ------------------------

typedef struct {
  size_t n;
  double rlist2[52];
  size_t nres;
  double res3la[3];
} gsl_compat_extrap_table;

inline void gsl_compat_initialise_table(gsl_compat_extrap_table *table) {
  table->n = 0;
  table->nres = 0;
}

inline void gsl_compat_append_table(gsl_compat_extrap_table *table, double y) {
  table->rlist2[table->n] = y;
  table->n++;
}

inline void gsl_compat_qelg(gsl_compat_extrap_table *table, double *result,
                            double *abserr) {
  double *epstab = table->rlist2;
  double *res3la = table->res3la;
  const size_t n = table->n - 1;

  const double current = epstab[n];

  double absolute = GSL_DBL_MAX;
  double relative = 5 * GSL_DBL_EPSILON * fabs(current);

  const size_t newelm = n / 2;
  const size_t n_orig = n;
  size_t n_final = n;
  const size_t nres_orig = table->nres;

  *result = current;
  *abserr = GSL_DBL_MAX;

  if (n < 2) {
    *abserr = GSL_MAX_DBL(absolute, relative);
    return;
  }

  epstab[n + 2] = epstab[n];
  epstab[n] = GSL_DBL_MAX;

  for (size_t i = 0; i < newelm; i++) {
    double res = epstab[n - 2 * i + 2];
    const double e0 = epstab[n - 2 * i - 2];
    const double e1 = epstab[n - 2 * i - 1];
    const double e2 = res;

    const double e1abs = fabs(e1);
    const double delta2 = e2 - e1;
    const double err2 = fabs(delta2);
    const double tol2 = GSL_MAX_DBL(fabs(e2), e1abs) * GSL_DBL_EPSILON;
    const double delta3 = e1 - e0;
    const double err3 = fabs(delta3);
    const double tol3 = GSL_MAX_DBL(e1abs, fabs(e0)) * GSL_DBL_EPSILON;

    if (err2 <= tol2 && err3 <= tol3) {
      // e0, e1 and e2 agree to machine accuracy: assume convergence
      *result = res;
      absolute = err2 + err3;
      relative = 5 * GSL_DBL_EPSILON * fabs(res);
      *abserr = GSL_MAX_DBL(absolute, relative);
      return;
    }

    const double e3 = epstab[n - 2 * i];
    epstab[n - 2 * i] = e1;
    const double delta1 = e1 - e3;
    const double err1 = fabs(delta1);
    const double tol1 = GSL_MAX_DBL(e1abs, fabs(e3)) * GSL_DBL_EPSILON;

    // Drop part of the table when two elements are nearly equal, or when the
    // table behaves irregularly
    if (err1 <= tol1 || err2 <= tol2 || err3 <= tol3) {
      n_final = 2 * i;
      break;
    }

    const double ss = (1 / delta1 + 1 / delta2) - 1 / delta3;

    if (fabs(ss * e1) <= 0.0001) {
      n_final = 2 * i;
      break;
    }

    res = e1 + 1 / ss;
    epstab[n - 2 * i] = res;

    const double error = err2 + fabs(res - e2) + err3;
    if (error <= *abserr) {
      *abserr = error;
      *result = res;
    }
  }

  const size_t limexp = 50 - 1;
  if (n_final == limexp) n_final = 2 * (limexp / 2);

  if (n_orig % 2 == 1) {
    for (size_t i = 0; i <= newelm; i++) epstab[1 + i * 2] = epstab[i * 2 + 3];
  } else {
    for (size_t i = 0; i <= newelm; i++) epstab[i * 2] = epstab[i * 2 + 2];
  }

  if (n_orig != n_final) {
    for (size_t i = 0; i <= n_final; i++)
      epstab[i] = epstab[n_orig - n_final + i];
  }

  table->n = n_final + 1;

  if (nres_orig < 3) {
    res3la[nres_orig] = *result;
    *abserr = GSL_DBL_MAX;
  } else {
    *abserr = (fabs(*result - res3la[2]) + fabs(*result - res3la[1]) +
               fabs(*result - res3la[0]));
    res3la[0] = res3la[1];
    res3la[1] = res3la[2];
    res3la[2] = *result;
  }

  table->nres = nres_orig + 1;

  *abserr = GSL_MAX_DBL(*abserr, 5 * GSL_DBL_EPSILON * fabs(*result));
}

//---------------- Adaptive integration with extrapolation -----------

inline int gsl_compat_positive(double result, double resabs) {
  return fabs(result) >= (1 - 50 * GSL_DBL_EPSILON) * resabs;
}

inline int gsl_compat_qags(const gsl_function *f, const double a, const double b,
                           const double epsabs, const double epsrel,
                           const size_t limit, gsl_integration_workspace *w,
                           double *result, double *abserr) {
  double result0, abserr0, resabs0, resasc0;
  double tolerance;
  double ertest = 0;
  double error_over_large_intervals = 0;
  double reseps = 0, abseps = 0, correc = 0;
  size_t ktmin = 0;
  int roundoff_type1 = 0, roundoff_type2 = 0, roundoff_type3 = 0;
  int error_type = 0, error_type2 = 0;
  int extrapolate = 0, disallow_extrapolation = 0;

  gsl_compat_extrap_table table;

  gsl_compat_initialise(w, a, b);

  *result = 0;
  *abserr = 0;

  if (limit > w->limit) return GSL_EINVAL;

  if (epsabs <= 0 && (epsrel < 50 * GSL_DBL_EPSILON || epsrel < 0.5e-28))
    return GSL_EBADTOL;

  gsl_integration_qk15(f, a, b, &result0, &abserr0, &resabs0, &resasc0);

  gsl_compat_set_initial_result(w, result0, abserr0);

  tolerance = GSL_MAX_DBL(epsabs, epsrel * fabs(result0));

  if (abserr0 <= 100 * GSL_DBL_EPSILON * resabs0 && abserr0 > tolerance) {
    *result = result0;
    *abserr = abserr0;
    return GSL_EROUND;
  } else if ((abserr0 <= tolerance && abserr0 != resasc0) || abserr0 == 0.0) {
    *result = result0;
    *abserr = abserr0;
    return GSL_SUCCESS;
  } else if (limit == 1) {
    *result = result0;
    *abserr = abserr0;
    return GSL_EMAXITER;
  }

  gsl_compat_initialise_table(&table);
  gsl_compat_append_table(&table, result0);

  double area = result0;
  double errsum = abserr0;
  double res_ext = result0;
  double err_ext = GSL_DBL_MAX;

  const int positive_integrand = gsl_compat_positive(result0, resabs0);

  size_t iteration = 1;
  int compute_result = 0;

  do {
    double a_i, b_i, r_i, e_i;
    double area1 = 0, area2 = 0;
    double error1 = 0, error2 = 0;
    double resabs1, resabs2, resasc1, resasc2;

    // Bisect the subinterval carrying the largest error estimate
    gsl_compat_retrieve(w, &a_i, &b_i, &r_i, &e_i);

    const size_t current_level = w->level[w->i] + 1;

    const double a1 = a_i;
    const double b1 = 0.5 * (a_i + b_i);
    const double a2 = b1;
    const double b2 = b_i;

    iteration++;

    gsl_integration_qk15(f, a1, b1, &area1, &error1, &resabs1, &resasc1);
    gsl_integration_qk15(f, a2, b2, &area2, &error2, &resabs2, &resasc2);

    const double area12 = area1 + area2;
    const double error12 = error1 + error2;
    const double last_e_i = e_i;

    errsum = errsum + error12 - e_i;
    area = area + area12 - r_i;

    tolerance = GSL_MAX_DBL(epsabs, epsrel * fabs(area));

    if (resasc1 != error1 && resasc2 != error2) {
      const double delta = r_i - area12;

      if (fabs(delta) <= 1.0e-5 * fabs(area12) && error12 >= 0.99 * e_i) {
        if (!extrapolate)
          roundoff_type1++;
        else
          roundoff_type2++;
      }
      if (iteration > 10 && error12 > e_i) roundoff_type3++;
    }

    if (roundoff_type1 + roundoff_type2 >= 10 || roundoff_type3 >= 20)
      error_type = 2;

    if (roundoff_type2 >= 5) error_type2 = 1;

    // Bad integrand behaviour at a point of the integration range
    if (gsl_compat_subinterval_too_small(a1, a2, b2)) error_type = 4;

    gsl_compat_update(w, a1, b1, area1, error1, a2, b2, area2, error2);

    if (errsum <= tolerance) {
      compute_result = 1;
      break;
    }

    if (error_type) break;

    if (iteration >= limit - 1) {
      error_type = 1;
      break;
    }

    if (iteration == 2) {
      error_over_large_intervals = errsum;
      ertest = tolerance;
      gsl_compat_append_table(&table, area);
      continue;
    }

    if (disallow_extrapolation) continue;

    error_over_large_intervals += -last_e_i;

    if (current_level < w->maximum_level)
      error_over_large_intervals += error12;

    if (!extrapolate) {
      // Extrapolate only once the interval to be bisected next is the
      // smallest one
      if (gsl_compat_large_interval(w)) continue;

      extrapolate = 1;
      w->nrmax = 1;
    }

    if (!error_type2 && error_over_large_intervals > ertest) {
      if (gsl_compat_increase_nrmax(w)) continue;
    }

    gsl_compat_append_table(&table, area);

    gsl_compat_qelg(&table, &reseps, &abseps);

    ktmin++;

    if (ktmin > 5 && err_ext < 0.001 * errsum) error_type = 5;

    if (abseps < err_ext) {
      ktmin = 0;
      err_ext = abseps;
      res_ext = reseps;
      correc = error_over_large_intervals;
      ertest = GSL_MAX_DBL(epsabs, epsrel * fabs(reseps));
      if (err_ext <= ertest) break;
    }

    if (table.n == 1) disallow_extrapolation = 1;

    if (error_type == 5) break;

    gsl_compat_reset_nrmax(w);
    extrapolate = 0;
    error_over_large_intervals = errsum;

  } while (iteration < limit);

  if (!compute_result) {
    *result = res_ext;
    *abserr = err_ext;

    if (err_ext == GSL_DBL_MAX) {
      compute_result = 1;
    } else if (error_type || error_type2) {
      if (error_type2) err_ext += correc;

      if (error_type == 0) error_type = 3;

      if (res_ext != 0.0 && area != 0.0) {
        if (err_ext / fabs(res_ext) > errsum / fabs(area)) compute_result = 1;
      } else if (err_ext > errsum) {
        compute_result = 1;
      }
    }

    if (!compute_result) {
      // Test on divergence
      const double max_area = GSL_MAX_DBL(fabs(res_ext), fabs(area));
      if (!(!positive_integrand && max_area < 0.01 * resabs0)) {
        const double ratio = res_ext / area;
        if (ratio < 0.01 || ratio > 100.0 || errsum > fabs(area))
          error_type = 6;
      }
    }
  }

  if (compute_result) {
    *result = gsl_compat_sum_results(w);
    *abserr = errsum;
  }

  if (error_type > 2) error_type--;

  switch (error_type) {
    case 0: return GSL_SUCCESS;
    case 1: return GSL_EMAXITER;
    case 2: return GSL_EROUND;
    case 3: return GSL_ESING;
    default: return GSL_EDIVERGE;
  }
}

typedef struct {
  double a;
  const gsl_function *f;
} gsl_compat_iu_params;

// Maps (a, +infinity) onto (0, 1] through x = a + (1 - t) / t
inline double gsl_compat_iu_transform(double t, void *params) {
  gsl_compat_iu_params *p = (gsl_compat_iu_params *)params;
  const double x = p->a + (1 - t) / t;
  const double y = GSL_FN_EVAL(p->f, x);
  return (y / t) / t;
}

inline int gsl_integration_qagiu(gsl_function *f, double a, double epsabs,
                                 double epsrel, size_t limit,
                                 gsl_integration_workspace *w, double *result,
                                 double *abserr) {
  gsl_compat_iu_params transform_params;
  transform_params.a = a;
  transform_params.f = f;

  gsl_function f_transform;
  f_transform.function = &gsl_compat_iu_transform;
  f_transform.params = &transform_params;

  return gsl_compat_qags(&f_transform, 0.0, 1.0, epsabs, epsrel, limit, w,
                         result, abserr);
}

#endif // GSL_COMPAT_H
