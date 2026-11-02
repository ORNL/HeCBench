// Isothermal, constant-density kinetics and one backward-Euler step for the
// detailed n-heptane size (560 species, 2500 reactions). SYCL port of
// src/stiff-ode-hip/kinetics.cuh with the same names and operation order.
//
// State variables are molar concentrations (mol/cm^3). Temperature is fixed
// for a zone, so Arrhenius coefficients and Kc are write-once. The step is
// BDF1 / backward Euler with an analytic Jacobian:
//   (I - dt J) δ = C0 - C + dt ω,   J = dω/dC
// solved by dense LU with partial pivoting. J is evaluated and factored at the
// start of each step and the factors are reused for the rest of its Newton
// iterations (modified Newton).
// The matrix lives in caller storage (global or local memory), never in a
// per-work-item private array.

#pragma once

#include "../stiff-ode-cuda/mechanism.h"

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>

constexpr double kRerg = 8.31446261815324e7;  // erg/mol/K
constexpr double kRcal = 1.98720425864083;    // cal/mol/K
constexpr double kPstd = 1.01325e6;           // dyne/cm^2 (1 atm, CHEMKIN standard state)
constexpr double kLn10 = 2.3025850929940457;
constexpr int kNewtonMax = 12;
constexpr double kNewtonTol = 1e-8;

struct Tables {
  const int* rtype;
  const int* reversible;
  const int* nreact;
  const int* nprod;
  const int* r_idx;
  const int* r_nu;
  const int* p_idx;
  const int* p_nu;
  const int* eff_off;
  const int* eff_len;
  const int* eff_sp;
  const double* eff_eps;
  const double* troe;
  const double* A_high;
  const double* B_high;
  const double* Ea_high;
  const double* A_low;
  const double* B_low;
  const double* Ea_low;
  const double* nasa_lo;
  const double* nasa_hi;
  const double* tmid;
};

// J (2.5 MB per zone) stays in global memory. The LU factors Width columns at
// a time in a local-memory tile of NS rows, and split assembly keeps
// kRxTerms doubles per reaction in global scratch. Tile rows are padded to an
// odd number of doubles so a sub-group walking a tile column hits distinct
// banks.
template <int Width>
constexpr int kTileLd = Width | 1;
template <int Width>
constexpr int kTileBytes = kTileLd<Width> * NS * (int)sizeof(double);
constexpr int kRxTerms = 10;

// exp for host code and for the device kernel.
inline double stiff_exp(double x) {
#if defined(__SYCL_DEVICE_ONLY__)
  return sycl::exp(x);
#else
  return std::exp(x);
#endif
}

// log for host code and for the device kernel.
inline double stiff_log(double x) {
#if defined(__SYCL_DEVICE_ONLY__)
  return sycl::log(x);
#else
  return std::log(x);
#endif
}

// fabs for host code and for the device kernel.
inline double stiff_fabs(double x) {
#if defined(__SYCL_DEVICE_ONLY__)
  return sycl::fabs(x);
#else
  return std::fabs(x);
#endif
}

// Arrhenius rate A T^b exp(-Ea/(R T)). Zero when A is not positive.
inline double arrhenius(double A, double b, double Ea, double T) {
  if (!(A > 0.0)) return 0.0;
  return stiff_exp(stiff_log(A) + b * stiff_log(T) - Ea / (kRcal * T));
}

// Dimensionless Gibbs energy G/RT from the NASA polynomial.
inline double species_G_RT(const Tables& tab, int sp, double T) {
  const double* a = (T <= tab.tmid[sp]) ? (tab.nasa_lo + sp * 7) : (tab.nasa_hi + sp * 7);
  double t2 = T * T;
  double t3 = t2 * T;
  double t4 = t3 * T;
  double Hrt = a[0] + a[1] * T / 2.0 + a[2] * t2 / 3.0 + a[3] * t3 / 4.0 + a[4] * t4 / 5.0 +
               a[5] / T;
  double Sr = a[0] * stiff_log(T) + a[1] * T + a[2] * t2 / 2.0 + a[3] * t3 / 3.0 + a[4] * t4 / 4.0 +
              a[6];
  return Hrt - Sr;
}

// log(Kc) with Kc in (mol/cm^3)^dn. dn excludes the third body.
inline double reaction_logKc(const Tables& tab, int r, double T) {
  double dG = 0.0;
  double dn = 0.0;
  for (int j = 0; j < tab.nreact[r]; ++j) {
    int sp = tab.r_idx[r * 3 + j];
    double nu = tab.r_nu[r * 3 + j];
    dG -= nu * species_G_RT(tab, sp, T);
    dn -= nu;
  }
  for (int j = 0; j < tab.nprod[r]; ++j) {
    int sp = tab.p_idx[r * 3 + j];
    double nu = tab.p_nu[r * 3 + j];
    dG += nu * species_G_RT(tab, sp, T);
    dn += nu;
  }
  return -dG + dn * stiff_log(kPstd / (kRerg * T));
}

// High- and low-pressure rates and log(Kc) for one reaction at temperature T.
inline void prepare_one(const Tables& tab, int r, double T, double& kf, double& k0, double& logKc) {
  kf = arrhenius(tab.A_high[r], tab.B_high[r], tab.Ea_high[r], T);
  k0 = arrhenius(tab.A_low[r], tab.B_low[r], tab.Ea_low[r], T);
  logKc = tab.reversible[r] ? reaction_logKc(tab, r, T) : 0.0;
}

// Net stoichiometric coefficient of species sp in reaction r.
inline int net_nu(const Tables& tab, int r, int sp) {
  int nu = 0;
  for (int j = 0; j < tab.nreact[r]; ++j) {
    if (tab.r_idx[r * 3 + j] == sp) nu -= tab.r_nu[r * 3 + j];
  }
  for (int j = 0; j < tab.nprod[r]; ++j) {
    if (tab.p_idx[r * 3 + j] == sp) nu += tab.p_nu[r * 3 + j];
  }
  return nu;
}

// Third-body concentration, including efficiency corrections.
template <int SpeciesCount = NS>
inline double third_body_M(const Tables& tab, int r, const double* C) {
  double M = 0.0;
  for (int i = 0; i < SpeciesCount; ++i) M += C[i];
  int off = tab.eff_off[r];
  int len = tab.eff_len[r];
  for (int e = 0; e < len; ++e) M += (tab.eff_eps[off + e] - 1.0) * C[tab.eff_sp[off + e]];
  return M;
}

// Troe falloff broadening factor.
inline double troe_F(const double* troe, double T, double Pr) {
  double a = troe[0];
  double T3 = troe[1];
  double T1 = troe[2];
  double T2 = troe[3];
  double Fcent = (1.0 - a) * stiff_exp(-T / T3) + a * stiff_exp(-T / T1) + stiff_exp(-T2 / T);
  if (Fcent < 1e-300) Fcent = 1e-300;
  double log10Fcent = stiff_log(Fcent) / kLn10;
  double c = -0.4 - 0.67 * log10Fcent;
  double n = 0.75 - 1.27 * log10Fcent;
  double log10Pr = stiff_log(Pr) / kLn10;
  double f1 = log10Pr + c;
  double den = n - 0.14 * f1;
  if (den > -1e-30 && den < 1e-30) den = (den < 0.0) ? -1e-30 : 1e-30;
  double f2 = f1 / den;
  double log10F = log10Fcent / (1.0 + f2 * f2);
  return stiff_exp(kLn10 * log10F);
}

// Effective first-order rate (Lindemann or Troe). Zero at M = 0.
inline double falloff_k(int rtype, double kinf, double k0, double M, const double* troe, double T) {
  if (!(kinf > 0.0) || !(k0 > 0.0) || !(M > 0.0)) return 0.0;
  double Pr = (k0 / kinf) * M;
  double k = kinf * (Pr / (1.0 + Pr));
  if (rtype == 3) k *= troe_F(troe, T, Pr);
  return k;
}

// Derivative of a falloff rate with respect to the third-body concentration.
inline double falloff_dk_dM(int rtype, double kinf, double k0, double M, const double* troe,
                            double T) {
  if (!(kinf > 0.0) || !(k0 > 0.0)) return 0.0;
  if (rtype == 2) {
    double Pr = (k0 / kinf) * M;
    double d = 1.0 + Pr;
    return k0 / (d * d);
  }
  double base = M > 0.0 ? M : 0.0;
  double h = (base > 0.0 ? base : 1e-30) * 1e-6;
  double Mp = base + h;
  double Mm = base - h;
  if (Mm < 0.0) Mm = 0.0;
  double denom = Mp - Mm;
  if (!(denom > 0.0)) return 0.0;
  double kp = falloff_k(rtype, kinf, k0, Mp, troe, T);
  double km = falloff_k(rtype, kinf, k0, Mm, troe, T);
  return (kp - km) / denom;
}

// prod = Π C^ν and d(prod)/dC for each slot. ν is 1 or 2.
inline void side_prod(const int* idx, const int* nu, int n, const double* C, double& prod,
                      double* dprod) {
  prod = 1.0;
  for (int j = 0; j < n; ++j) {
    double c = C[idx[j]];
    prod *= (nu[j] == 1) ? c : c * c;
  }
  for (int j = 0; j < n; ++j) {
    double others = 1.0;
    for (int i = 0; i < n; ++i) {
      if (i == j) continue;
      double c = C[idx[i]];
      others *= (nu[i] == 1) ? c : c * c;
    }
    dprod[j] = (nu[j] == 1) ? others : 2.0 * C[idx[j]] * others;
  }
}

// Per-reaction rate terms, independent of the species row. kJac=false skips
// the falloff derivative; k_f, k_r, and the products do not depend on it.
struct RxTerms {
  int rt, nf, np;
  double k_f, k_r, dkf_dM, dkr_dM;
  double prod_f, prod_r;
  double dpf[3], dpr[3];
};

// Evaluates reaction r into t.
template <int SpeciesCount = NS, bool kJac = true>
inline void reaction_terms(const Tables& tab, int r, const double* C, const double* kf,
                           const double* k0, const double* logKc, double T, RxTerms& t) {
  int rt = tab.rtype[r];
  double kinf = kf[r];
  double klow = k0[r];
  double M = 0.0;
  double k_f = 0.0;
  double dkf_dM = 0.0;
  const double* tr = tab.troe + r * 4;
  if (rt == 0) {
    k_f = kinf;
  } else if (rt == 1) {
    M = third_body_M<SpeciesCount>(tab, r, C);
    k_f = kinf * M;
    dkf_dM = kinf;
  } else {
    M = third_body_M<SpeciesCount>(tab, r, C);
    k_f = falloff_k(rt, kinf, klow, M, tr, T);
    if constexpr (kJac) dkf_dM = falloff_dk_dM(rt, kinf, klow, M, tr, T);
  }

  double k_r = 0.0;
  double dkr_dM = 0.0;
  if (tab.reversible[r]) {
    double lkc = logKc[r];
    if (k_f > 0.0) {
      double log_kr = stiff_log(k_f) - lkc;
      k_r = (log_kr > -700.0 && log_kr < 700.0) ? stiff_exp(log_kr)
                                                 : (log_kr >= 700.0 ? stiff_exp(700.0) : 0.0);
    }
    if (kJac && dkf_dM != 0.0) {
      double sign = dkf_dM > 0.0 ? 1.0 : -1.0;
      double log_d = stiff_log(stiff_fabs(dkf_dM)) - lkc;
      double mag = (log_d > -700.0 && log_d < 700.0) ? stiff_exp(log_d)
                                                      : (log_d >= 700.0 ? stiff_exp(700.0) : 0.0);
      dkr_dM = sign * mag;
    }
  }

  t.rt = rt;
  t.nf = tab.nreact[r];
  t.np = tab.nprod[r];
  t.prod_r = 0.0;
  for (int j = 0; j < 3; ++j) t.dpr[j] = 0.0;
  side_prod(tab.r_idx + r * 3, tab.r_nu + r * 3, t.nf, C, t.prod_f, t.dpf);
  bool need_rev = tab.reversible[r] && (k_r != 0.0 || dkr_dM != 0.0);
  if (need_rev) side_prod(tab.p_idx + r * 3, tab.p_nu + r * 3, t.np, C, t.prod_r, t.dpr);
  t.k_f = k_f;
  t.k_r = k_r;
  t.dkf_dM = dkf_dM;
  t.dkr_dM = dkr_dM;
}

// Add reaction r into species row i. Jrow has NS entries. No-op if ν_i = 0.
// kJac=false adds only the rate; ω is bit-identical to the kJac=true value.
template <int SpeciesCount = NS, bool kJac = true>
inline void reaction_row_update(const Tables& tab, int r, int i, const double* C, const double* kf,
                                const double* k0, const double* logKc, double T, double* Jrow,
                                double& wi) {
  int nu_i = net_nu(tab, r, i);
  if (nu_i == 0) return;
  double nu = nu_i;
  RxTerms t;
  reaction_terms<SpeciesCount, kJac>(tab, r, C, kf, k0, logKc, T, t);
  wi += nu * (t.k_f * t.prod_f - t.k_r * t.prod_r);
  if constexpr (kJac) {
    for (int j = 0; j < t.nf; ++j) Jrow[tab.r_idx[r * 3 + j]] += nu * t.k_f * t.dpf[j];
    if (t.k_r != 0.0) {
      for (int j = 0; j < t.np; ++j) Jrow[tab.p_idx[r * 3 + j]] -= nu * t.k_r * t.dpr[j];
    }
    if (t.rt != 0) {
      double scale = nu * (t.dkf_dM * t.prod_f - t.dkr_dM * t.prod_r);
      for (int s = 0; s < SpeciesCount; ++s) Jrow[s] += scale;
      int off = tab.eff_off[r];
      int len = tab.eff_len[r];
      for (int e = 0; e < len; ++e) Jrow[tab.eff_sp[off + e]] += scale * (tab.eff_eps[off + e] - 1.0);
    }
  }
}

// kJac=false leaves J untouched, so a factorization stored there survives.
template <int SpeciesCount = NS, bool kJac = true>
inline void assemble_row(const Tables& tab, int i, const double* C, const double* kf,
                         const double* k0, const double* logKc, double T, const int* row_off,
                         const int* row_reac, double* J, double* w) {
  double* Jrow = J + i * SpeciesCount;
  if constexpr (kJac) {
    for (int k = 0; k < SpeciesCount; ++k) Jrow[k] = 0.0;
  }
  w[i] = 0.0;
  for (int p = row_off[i]; p < row_off[i + 1]; ++p) {
    reaction_row_update<SpeciesCount, kJac>(tab, row_reac[p], i, C, kf, k0, logKc, T, Jrow, w[i]);
  }
}

// In-place LU with partial pivoting. A row interchange swaps whole rows, so the
// multipliers below the diagonal travel with their row, and perm[i] is the
// original row now at position i. Returns false on a pivot at or below 1e-20.
template <int SpeciesCount = NS>
inline bool lu_factor(double* A, int* perm) {
  constexpr int n = SpeciesCount;
  for (int i = 0; i < n; ++i) perm[i] = i;
  for (int col = 0; col < n; ++col) {
    int piv = col;
    double maxa = stiff_fabs(A[col * n + col]);
    for (int row = col + 1; row < n; ++row) {
      double v = stiff_fabs(A[row * n + col]);
      if (v > maxa) {
        maxa = v;
        piv = row;
      }
    }
    if (!(maxa > 1e-20)) return false;
    if (piv != col) {
      for (int k = 0; k < n; ++k) {
        double tmp = A[col * n + k];
        A[col * n + k] = A[piv * n + k];
        A[piv * n + k] = tmp;
      }
      int t = perm[col];
      perm[col] = perm[piv];
      perm[piv] = t;
    }
    double diag = A[col * n + col];
    for (int row = col + 1; row < n; ++row) {
      double f = A[row * n + col] / diag;
      A[row * n + col] = f;
      for (int k = col + 1; k < n; ++k) A[row * n + k] -= f * A[col * n + k];
    }
  }
  return true;
}

// b is already in pivot order. Both sweeps are column-oriented, so each entry
// of b sees its updates in the same order as newton_block.
template <int SpeciesCount = NS>
inline void lu_solve(const double* A, double* b) {
  constexpr int n = SpeciesCount;
  for (int col = 0; col + 1 < n; ++col) {
    double bc = b[col];
    for (int row = col + 1; row < n; ++row) b[row] -= A[row * n + col] * bc;
  }
  for (int row = n - 1; row >= 0; --row) {
    double x = b[row] / A[row * n + row];
    b[row] = x;
    for (int i = 0; i < row; ++i) b[i] -= A[i * n + row] * x;
  }
}

// One Newton update of the backward-Euler step. factor=true overwrites J with
// the LU factors of I - dt J. Later iterations of the same step pass
// factor=false and reuse them (modified Newton), so only ω is reassembled.
// Returns the max relative accepted update (NaN if any update is NaN), or -1
// if the factorization failed.
template <int SpeciesCount = NS>
inline double apply_newton(double* C, const double* C0, const double* w, double* J, double* rhs,
                           int* perm, double dt, bool factor) {
  constexpr int n = SpeciesCount;
  if (factor) {
    for (int i = 0; i < n; ++i) {
      for (int k = 0; k < n; ++k) {
        double a = -dt * J[i * n + k];
        if (i == k) a += 1.0;
        J[i * n + k] = a;
      }
    }
    if (!lu_factor<SpeciesCount>(J, perm)) return -1.0;
  }
  for (int i = 0; i < n; ++i) {
    int p = perm[i];
    rhs[i] = C0[p] - C[p] + dt * w[p];
  }
  lu_solve<SpeciesCount>(J, rhs);
  double rel = 0.0;
  for (int i = 0; i < n; ++i) {
    double old = C[i];
    double nxt = old + rhs[i];
    if (nxt < 0.0) nxt = 0.0;
    C[i] = nxt;
    double change = stiff_fabs(nxt - old);
    double scale = stiff_fabs(old) + stiff_fabs(nxt) + 1e-20;
    double r = change / scale;
    if (r > rel || r != r) rel = r;
  }
  return rel;
}

// Work-group barrier.
inline void block_sync(sycl::nd_item<1> item) {
  sycl::group_barrier(item.get_group());
}

// Sub-group-0 maximum of v[0..n), starting from 0 like the serial loop. A NaN
// wins, so a NaN update cannot pass for convergence.
template <int SpeciesCount, int Ws>
inline double max_wave(sycl::sub_group sg, const double* v) {
  constexpr int ws = Ws;
  const int lane = sg.get_local_linear_id();
  double m = 0.0;
  for (int i = lane; i < SpeciesCount; i += ws)
    if (v[i] > m || v[i] != v[i]) m = v[i];
  for (int off = ws >> 1; off > 0; off >>= 1) {
    double o = sycl::permute_group_by_xor(sg, m, off);
    if (o > m || o != o) m = o;
  }
  return m;
}

// Lane l's value of v. l is sub-group-uniform, so AMD reads it with readlane.
inline double lane_value(sycl::sub_group sg, double v, int l) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__)
  (void)sg;
  const std::uint64_t u = sycl::bit_cast<std::uint64_t>(v);
  const int lo = __builtin_amdgcn_readlane((int)(std::uint32_t)u, l);
  const int hi = __builtin_amdgcn_readlane((int)(std::uint32_t)(u >> 32), l);
  return sycl::bit_cast<double>(((std::uint64_t)(std::uint32_t)hi << 32) | (std::uint32_t)lo);
#else
  return sycl::select_from_group(sg, v, l);
#endif
}

// Marks a value that is equal on every lane of the sub-group.
inline int wave_uniform(int v) {
#if defined(__SYCL_DEVICE_ONLY__) && defined(__AMDGCN__)
  return __builtin_amdgcn_readfirstlane(v);
#else
  return v;
#endif
}

// lu_solve across the work-group, blocked by kSolveRows rows (at most one
// sub-group). x lives in sx (the local-memory tile, idle once J is factored)
// and each diagonal block is staged in sd. Sub-group 0 substitutes the
// diagonal block; then every work-item takes one row outside it and applies
// the block's columns in order. Every entry sees the same updates in the same
// order as lu_solve.
template <int Ws>
constexpr int kSolveRows = Ws < 32 ? Ws : 32;

template <int SpeciesCount, int Ws>
inline void lu_solve_block(sycl::nd_item<1> item, const double* A, const double* b, double* out,
                           double* sx, double* sd) {
  constexpr int n = SpeciesCount;
  constexpr int nb = kSolveRows<Ws>;
  const int tid = item.get_local_id(0);
  const int stride = item.get_local_range(0);
  sycl::sub_group sg = item.get_sub_group();
  const int lane = tid % nb;
  for (int i = tid; i < n; i += stride) sx[i] = b[i];
  for (int c0 = 0; c0 < n; c0 += nb) {
    const int bw = n - c0 < nb ? n - c0 : nb;
    const int c1 = c0 + bw;
    for (int e = tid; e < bw * bw; e += stride) {
      const int q = e / bw;
      const int j = e - q * bw;
      sd[e] = A[(c0 + q) * n + c0 + j];
    }
    block_sync(item);
    if (tid < nb) {
      double xv = lane < bw ? sx[c0 + lane] : 0.0;
      for (int j = 0; j + 1 < bw; ++j) {
        const double bc = lane_value(sg, xv, j);
        if (lane > j && lane < bw) xv -= sd[lane * bw + j] * bc;
      }
      if (lane < bw) sx[c0 + lane] = xv;
    }
    block_sync(item);
    for (int r = c1 + tid; r < n; r += stride) {
      const double* Ar = A + r * n;
      double xr = sx[r];
      for (int col = c0; col < c1; ++col) xr -= Ar[col] * sx[col];
      sx[r] = xr;
    }
  }
  for (int c1 = n; c1 > 0; c1 -= nb) {
    const int bw = c1 < nb ? c1 : nb;
    const int c0 = c1 - bw;
    for (int e = tid; e < bw * bw; e += stride) {
      const int q = e / bw;
      const int j = e - q * bw;
      sd[e] = A[(c0 + q) * n + c0 + j];
    }
    block_sync(item);
    if (tid < nb) {
      double xv = lane < bw ? sx[c0 + lane] : 0.0;
      for (int j = bw - 1; j >= 0; --j) {
        const double xr = lane_value(sg, xv, j) / sd[j * bw + j];
        if (lane == j) xv = xr;
        if (lane < j) xv -= sd[lane * bw + j] * xr;
      }
      if (lane < bw) sx[c0 + lane] = xv;
    }
    block_sync(item);
    for (int r = tid; r < c0; r += stride) {
      const double* Ar = A + r * n;
      double xr = sx[r];
      for (int col = c1 - 1; col >= c0; --col) xr -= Ar[col] * sx[col];
      sx[r] = xr;
    }
  }
  block_sync(item);
  for (int i = tid; i < n; i += stride) out[i] = sx[i];
}

// Rows rb..re-1, columns c0..c0+w-1 of A into the tile at stride ld.
template <int SpeciesCount>
inline void stage_rows(sycl::nd_item<1> item, const double* A, double* tile, int ld, int rb,
                       int re, int c0, int w) {
  constexpr int n = SpeciesCount;
  const int tid = item.get_local_id(0);
  const int stride = item.get_local_range(0);
  const int tot = (re - rb) * w;
  for (int e = tid; e < tot; e += stride) {
    const int q = e / w;
    const int j = e - q * w;
    tile[q * ld + j] = A[(rb + q) * n + c0 + j];
  }
}

// Columns kb..ke-1 take the factored columns p0..p0+pw-1: rows p0..p0+pw-1
// become U (row p0+i taking the updates of columns p0..p0+i-1 in order) and
// every row below takes all pw updates in column order. Each work-item owns
// one column and a stride of G rows and keeps the column's U entries in
// registers. Without kStaged the tile holds L rows p0..n-1 at stride ld; with
// it, L is staged through the tile in chunks of cap / ld rows.
template <int SpeciesCount, int Width, bool kStaged>
inline void update_cols(sycl::nd_item<1> item, double* A, double* tile, int ld, int cap, int p0,
                        int pw, int kb, int ke) {
  constexpr int n = SpeciesCount;
  const int tid = item.get_local_id(0);
  const int stride = item.get_local_range(0);
  const int W = ke - kb;
  const int p1 = p0 + pw;
  const int R = kStaged ? cap / ld : n;
  // With stride >= W every work-item takes part, the first `extra` columns
  // getting one more group than the rest; otherwise work-group-uniform rounds
  // of stride columns take one group each.
  const int base = stride >= W ? stride / W : 1;
  const int extra = stride >= W ? stride - base * W : 0;
  const int nthr = stride >= W ? stride : W;
  for (int t0 = 0; t0 < nthr; t0 += stride) {
    const int t = t0 + tid;
    const bool active = t < nthr;
    int kk, g, G;
    if (t < (base + 1) * extra) {
      kk = t % extra;
      g = t / extra;
      G = base + 1;
    } else {
      const int t1 = t - (base + 1) * extra;
      kk = extra + t1 % (W - extra);
      g = t1 / (W - extra);
      G = base;
    }
    const int col = kb + kk;
    int rb = p0;
    int re = kStaged ? (p0 + R < n ? p0 + R : n) : n;
    if (kStaged) {
      block_sync(item);
      stage_rows<n>(item, A, tile, ld, rb, re, p0, pw);
      block_sync(item);
    }
    double u[Width];
#pragma unroll
    for (int i = 0; i < Width; ++i) {
      if (active && i < pw) {
        double a = A[(p0 + i) * n + col];
#pragma unroll
        for (int c = 0; c < i; ++c) a -= tile[i * ld + c] * u[c];
        u[i] = a;
      } else {
        u[i] = 0.0;
      }
    }
    block_sync(item);
    if (active && g == 0) {
#pragma unroll
      for (int i = 1; i < Width; ++i)
        if (i < pw) A[(p0 + i) * n + col] = u[i];
    }
    int r = p1 + g;
    for (;;) {
      if (active) {
        // Four independent rows hide the global-load latency. Each row still
        // subtracts columns p0..p1-1 in order.
        for (; r + 3 * G < re; r += 4 * G) {
          const double* f0 = tile + (r - rb) * ld;
          const double* f1 = tile + (r + G - rb) * ld;
          const double* f2 = tile + (r + 2 * G - rb) * ld;
          const double* f3 = tile + (r + 3 * G - rb) * ld;
          double a0 = A[r * n + col];
          double a1 = A[(r + G) * n + col];
          double a2 = A[(r + 2 * G) * n + col];
          double a3 = A[(r + 3 * G) * n + col];
#pragma unroll
          for (int c = 0; c < Width; ++c) {
            if (c < pw) {
              double uc = u[c];
              a0 -= f0[c] * uc;
              a1 -= f1[c] * uc;
              a2 -= f2[c] * uc;
              a3 -= f3[c] * uc;
            }
          }
          A[r * n + col] = a0;
          A[(r + G) * n + col] = a1;
          A[(r + 2 * G) * n + col] = a2;
          A[(r + 3 * G) * n + col] = a3;
        }
        for (; r < re; r += G) {
          const double* fr = tile + (r - rb) * ld;
          double a = A[r * n + col];
#pragma unroll
          for (int c = 0; c < Width; ++c)
            if (c < pw) a -= fr[c] * u[c];
          A[r * n + col] = a;
        }
      }
      if (!kStaged || re >= n) break;
      rb = re;
      re = rb + R < n ? rb + R : n;
      block_sync(item);
      stage_rows<n>(item, A, tile, ld, rb, re, p0, pw);
      block_sync(item);
    }
  }
  block_sync(item);
}

// lu_factor across the work-group with A in global memory, blocked by Panel
// columns. Each panel is factored Sub columns at a time in the local-memory
// tile (rows s0..n-1, one work-item per row), and each sub-panel's columns
// are applied at once to the rest of the panel. Then the columns right of the
// panel take all Panel updates in one pass (update_cols), so a trailing entry
// is read and written once per panel instead of once per column. Every entry
// gets the same subtractions in the same order as lu_factor, so the factors
// are bit-identical.
template <int SpeciesCount, int Ws, int Panel, int Sub>
inline void lu_factor_tiled(sycl::nd_item<1> item, double* A, int* perm, double* tile, int* spiv,
                            int* sfail) {
  constexpr int n = SpeciesCount;
  constexpr int ld = kTileLd<Sub>;
  const int tid = item.get_local_id(0);
  const int stride = item.get_local_range(0);
  sycl::sub_group sg = item.get_sub_group();
  constexpr int ws = Ws;
  for (int c0 = 0; c0 < n; c0 += Panel) {
    const int c1 = c0 + Panel < n ? c0 + Panel : n;
    for (int s0 = c0; s0 < c1; s0 += Sub) {
      const int bw = c1 - s0 < Sub ? c1 - s0 : Sub;
      const int s1 = s0 + bw;
      const int prow = n - s0;
      for (int e = tid; e < prow * bw; e += stride) {
        const int q = e / bw;
        const int j = e - q * bw;
        tile[q * ld + j] = A[(s0 + q) * n + s0 + j];
      }
      block_sync(item);
      for (int col = s0; col < s1; ++col) {
        const int lc = col - s0;
        if (tid < ws) {
          const int lane = tid;
          double best = -1.0;
          int bi = n;
          for (int r = col + lane; r < n; r += ws) {
            double v = stiff_fabs(tile[(r - s0) * ld + lc]);
            if (v > best) {
              best = v;
              bi = r;
            }
          }
          for (int off = ws >> 1; off > 0; off >>= 1) {
            double ov = sycl::permute_group_by_xor(sg, best, off);
            int oi = sycl::permute_group_by_xor(sg, bi, off);
            if (ov > best || (ov == best && oi < bi)) {
              best = ov;
              bi = oi;
            }
          }
          if (lane == 0) {
            double d = stiff_fabs(tile[lc * ld + lc]);
            if (d != d || !(best > 1e-20)) {
              *sfail = 1;
            } else {
              *spiv = bi;
              if (bi != col) {
                int t = perm[col];
                perm[col] = perm[bi];
                perm[bi] = t;
              }
            }
          }
        }
        block_sync(item);
        if (*sfail) return;
        const int piv = *spiv;
        if (piv != col) {
          for (int k = tid; k < n; k += stride) {
            if (k >= s0 && k < s1) {
              const int j = k - s0;
              double tmp = tile[lc * ld + j];
              tile[lc * ld + j] = tile[(piv - s0) * ld + j];
              tile[(piv - s0) * ld + j] = tmp;
            } else {
              double tmp = A[col * n + k];
              A[col * n + k] = A[piv * n + k];
              A[piv * n + k] = tmp;
            }
          }
          block_sync(item);
        }
        // Four pivot-row loads ahead of the stores, since tr may alias pr.
        const double* pr = tile + lc * ld;
        const double diag = pr[lc];
        for (int r = col + 1 + tid; r < n; r += stride) {
          double* tr = tile + (r - s0) * ld;
          const double f = tr[lc] / diag;
          int j = lc + 1;
          for (; j + 3 < bw; j += 4) {
            const double p0 = pr[j], p1 = pr[j + 1], p2 = pr[j + 2], p3 = pr[j + 3];
            const double t0 = tr[j], t1 = tr[j + 1], t2 = tr[j + 2], t3 = tr[j + 3];
            tr[j] = t0 - f * p0;
            tr[j + 1] = t1 - f * p1;
            tr[j + 2] = t2 - f * p2;
            tr[j + 3] = t3 - f * p3;
          }
          for (; j < bw; ++j) tr[j] -= f * pr[j];
          tr[lc] = f;
        }
        block_sync(item);
      }
      for (int e = tid; e < prow * bw; e += stride) {
        const int q = e / bw;
        const int j = e - q * bw;
        A[(s0 + q) * n + s0 + j] = tile[q * ld + j];
      }
      block_sync(item);
      if (s1 < c1) {
        update_cols<n, Sub, false>(item, A, tile, ld, n * ld, s0, bw, s1, c1);
      } else if constexpr (Sub == Panel) {
        if (c1 < n) update_cols<n, Sub, false>(item, A, tile, ld, n * ld, s0, bw, c1, n);
      }
    }
    if constexpr (Sub != Panel) {
      if (c1 < n)
        update_cols<n, Panel, true>(item, A, tile, kTileLd<Panel>, n * ld, c0, c1 - c0, c1, n);
    }
  }
}

// apply_newton across the work-group. w holds ω on entry and the update on
// exit. J holds I - dt J from split_assemble and is factored in panels by
// lu_factor_tiled, then substituted by lu_solve_block.
template <int SpeciesCount, int Ws, int Panel, int Sub>
inline void newton_block(sycl::nd_item<1> item, double* C, const double* C0, double* w, double* J,
                         double* rhs, int* perm, double dt, bool factor, double* tile,
                         double* srel, int* sfail, int* spiv) {
  constexpr int n = SpeciesCount;
  const int tid = item.get_local_id(0);
  const int stride = item.get_local_range(0);
  if (factor) {
    for (int i = tid; i < n; i += stride) perm[i] = i;
    block_sync(item);
    lu_factor_tiled<n, Ws, Panel, Sub>(item, J, perm, tile, spiv, sfail);
    if (*sfail) {
      if (tid == 0) *srel = -1.0;
      block_sync(item);
      return;
    }
  }
  for (int i = tid; i < n; i += stride) {
    int p = perm[i];
    rhs[i] = C0[p] - C[p] + dt * w[p];
  }
  block_sync(item);
  lu_solve_block<n, Ws>(item, J, rhs, w, tile, tile + n);
  block_sync(item);
  for (int i = tid; i < n; i += stride) {
    double old = C[i];
    double nxt = old + w[i];
    if (nxt < 0.0) nxt = 0.0;
    C[i] = nxt;
    rhs[i] = stiff_fabs(nxt - old) / (stiff_fabs(old) + stiff_fabs(nxt) + 1e-20);
  }
  block_sync(item);
  sycl::sub_group sg = item.get_sub_group();
  if (tid < Ws) {
    double m = max_wave<n, Ws>(sg, rhs);
    if (tid == 0) *srel = m;
  }
  block_sync(item);
}

// Split assembly. Phase 1 gives each reaction to one work-item and stores its
// terms in rx (structure of arrays: k_f, k_r, F, G, dpf[3], dpr[3]). Phase 2
// gives each species row to one sub-group and each column (and ω, column n) to
// one lane. A lane walks the row's reactions in CSR order and adds exactly the
// terms reaction_row_update adds to its entry, so J and ω are bit-identical to
// assemble_row, and then stores the row of I - dt J. Sub-groups take rows from
// a local counter (snext), since rows differ widely in cost. factor=false
// stores only F and sums ω one work-item per row.
template <int SpeciesCount, int ReactionCount, int Ws>
inline void split_assemble(sycl::nd_item<1> item, const Tables& tab, const double* C,
                           const double* kf, const double* k0, const double* logKc, double T,
                           const int* row_off, const int* row_reac, double* rx, double* J,
                           double* w, bool factor, double dt, int* snext) {
  constexpr int n = SpeciesCount;
  constexpr int nr = ReactionCount;
  const int tid = item.get_local_id(0);
  const int bdim = item.get_local_range(0);
  double* kF = rx;
  double* kR = rx + nr;
  double* F = rx + 2 * nr;
  double* G = rx + 3 * nr;
  double* dpf = rx + 4 * nr;
  double* dpr = rx + 7 * nr;
  for (int r = tid; r < nr; r += bdim) {
    RxTerms t;
    if (factor) {
      reaction_terms<n, true>(tab, r, C, kf, k0, logKc, T, t);
      kF[r] = t.k_f;
      kR[r] = t.k_r;
      G[r] = t.dkf_dM * t.prod_f - t.dkr_dM * t.prod_r;
      for (int j = 0; j < 3; ++j) {
        dpf[j * nr + r] = j < t.nf ? t.dpf[j] : 0.0;
        dpr[j * nr + r] = t.dpr[j];
      }
    } else {
      reaction_terms<n, false>(tab, r, C, kf, k0, logKc, T, t);
    }
    F[r] = t.k_f * t.prod_f - t.k_r * t.prod_r;
  }
  if (tid == 0) *snext = bdim / Ws;
  block_sync(item);
  if (!factor) {
    for (int i = tid; i < n; i += bdim) {
      double wi = 0.0;
      for (int p = row_off[i]; p < row_off[i + 1]; ++p) {
        const int r = row_reac[p];
        wi += (double)net_nu(tab, r, i) * F[r];
      }
      w[i] = wi;
    }
    return;
  }
  sycl::sub_group sg = item.get_sub_group();
  constexpr int ws = Ws;
  const int lane = sg.get_local_linear_id();
  // Lane l accumulates the columns k with k % ws == l of the row in place
  // in J, and the lane of column n accumulates ω.
  for (int i = wave_uniform(tid / ws); i < n;) {
    const int pb = row_off[i];
    const int pe = row_off[i + 1];
    double* Jrow = J + i * n;
    double wi = 0.0;
    for (int k = lane; k < n; k += ws) Jrow[k] = 0.0;
    for (int p = pb; p < pe; ++p) {
      const int r = row_reac[p];
      const double nu = (double)net_nu(tab, r, i);
      if ((n % ws) == lane) wi += nu * F[r];
      const double tf = nu * kF[r];
      const int nf = tab.nreact[r];
      for (int j = 0; j < nf; ++j) {
        const int sp = tab.r_idx[r * 3 + j];
        if ((sp % ws) == lane) Jrow[sp] += tf * dpf[j * nr + r];
      }
      const double kr = kR[r];
      if (kr != 0.0) {
        const double tr = nu * kr;
        const int np = tab.nprod[r];
        for (int j = 0; j < np; ++j) {
          const int sp = tab.p_idx[r * 3 + j];
          if ((sp % ws) == lane) Jrow[sp] -= tr * dpr[j * nr + r];
        }
      }
      if (tab.rtype[r] != 0) {
        const double scale = nu * G[r];
        const int off = tab.eff_off[r];
        const int len = tab.eff_len[r];
        for (int k = lane; k < n; k += ws) {
          Jrow[k] += scale;
          for (int e = 0; e < len; ++e)
            if (tab.eff_sp[off + e] == k) Jrow[k] += scale * (tab.eff_eps[off + e] - 1.0);
        }
      }
    }
    if ((n % ws) == lane) w[i] = wi;
    for (int k = lane; k < n; k += ws) {
      double a = -dt * Jrow[k];
      if (k == i) a += 1.0;
      Jrow[k] = a;
    }
    int next = 0;
    if (lane == 0) {
      sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                       sycl::access::address_space::local_space>
          counter(*snext);
      next = counter.fetch_add(1);
    }
    i = wave_uniform(sycl::group_broadcast(sg, next));
  }
}

// Kernel shape per sub-group width. Block bounds the registers per work-item,
// the trailing update holds Panel U entries in registers, and the tile holds
// Sub columns. 32-wide sub-groups run 512 work-items with 40-column panels in
// 8-column tiles; 64-wide ones run 1024 with 24-column panels in 12-column
// tiles.
template <int Ws>
struct SolverShapeFor {
  static constexpr int block = 512;
  static constexpr int panel = 40;
  static constexpr int sub = 8;
};
template <>
struct SolverShapeFor<64> {
  static constexpr int block = 1024;
  static constexpr int panel = 24;
  static constexpr int sub = 12;
};

// One zone per work-group: split assembly with its terms in gPart, the panel
// LU through the local-memory tile, and the blocked substitution. Everything
// else is in global memory. The first Newton iteration of a step assembles J
// and factors it; the rest reassemble ω and reuse the factors. Ws is the
// sub-group width the kernel is compiled for.
template <int Ws>
inline void solver_kernel(sycl::nd_item<1> item, double* tile, double* srel, int* sfail, int* spiv,
                          int* snext, Tables tab, const int* row_off, const int* row_reac, const double* Tzone,
                          double* gC, double* gC0, double* gW, double* gRhs, double* gJ,
                          int* gPerm, double* gKf, double* gK0, double* gLogKc, int steps,
                          double dt, int* status, int* niters, double* gPart) {
  const int zone = item.get_group(0);
  const int tid = item.get_local_id(0);
  const int bdim = item.get_local_range(0);
  double* C = gC + zone * NS;
  double* C0 = gC0 + zone * NS;
  double* w = gW + zone * NS;
  double* rhs = gRhs + zone * NS;
  double* J = gJ + (size_t)zone * NS * NS;
  int* perm = gPerm + zone * NS;
  double* kf = gKf + zone * NR;
  double* k0 = gK0 + zone * NR;
  double* logKc = gLogKc + zone * NR;
  double* rx = gPart + (size_t)zone * kRxTerms * NR;

  const double T = Tzone[zone];
  for (int r = tid; r < NR; r += bdim) {
    prepare_one(tab, r, T, kf[r], k0[r], logKc[r]);
  }
  block_sync(item);

  if (tid == 0) {
    *srel = 1.0;
    *sfail = 0;
  }
  block_sync(item);

  for (int step = 0; step < steps; ++step) {
    if (*sfail) break;
    for (int i = tid; i < NS; i += bdim) C0[i] = C[i];
    block_sync(item);
    for (int it = 0; it < kNewtonMax; ++it) {
      const bool factor = it == 0;
      split_assemble<NS, NR, Ws>(item, tab, C, kf, k0, logKc, T, row_off, row_reac, rx, J, w,
                                 factor, dt, snext);
      block_sync(item);
      newton_block<NS, Ws, SolverShapeFor<Ws>::panel, SolverShapeFor<Ws>::sub>(
          item, C, C0, w, J, rhs, perm, dt, factor, tile, srel, sfail, spiv);
      block_sync(item);
      if (tid == 0) {
        int done = it + 1;
        if (done > niters[zone]) niters[zone] = done;
      }
      block_sync(item);
      if (*sfail || *srel < kNewtonTol) break;
    }
    if (*sfail) {
      if (tid == 0) status[zone] = 1;
      break;
    }
    if (!(*srel < kNewtonTol)) {
      if (tid == 0) status[zone] = 2;
      break;
    }
  }
}

template <int Ws>
class solver_kernel_name;

// Kernel properties. The work-group size bound is the shape's block; without it
// NVIDIA register allocation assumes fewer registers per work-item than the
// trailing update needs. The sub-group width is fixed per instantiation so
// lane arithmetic and the per-lane register arrays are sized at compile time.
template <int Ws, typename F>
struct SolverBody {
  F f;
  // Runs the kernel body for one work-item.
  void operator()(sycl::nd_item<1> item) const { f(item); }
  // Requests the shape's work-group size and this sub-group width.
  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    namespace syclex = sycl::ext::oneapi::experimental;
    return syclex::properties{syclex::max_work_group_size<SolverShapeFor<Ws>::block>,
                              syclex::sub_group_size<Ws>};
  }
};

// One work-group per zone, with the shape's tile of local memory (the HIP
// dynamic LDS); srel, sfail, spiv, and snext are the static __shared__ scalars
// of the HIP kernel.
template <int Ws>
sycl::event submit_solver_ws(sycl::queue& q, int zones, int blk, Tables tab, const int* row_off,
                             const int* row_reac, const double* Tzone, double* gC, double* gC0,
                             double* gW, double* gRhs, double* gJ, int* gPerm, double* gKf,
                             double* gK0, double* gLogKc, int steps, double dt, int* status,
                             int* niters, double* gPart) {
  const auto ndr = sycl::nd_range<1>(sycl::range<1>((size_t)zones * (size_t)blk),
                                     sycl::range<1>((size_t)blk));
  // Allocates the local tile and scalars, then launches the kernel.
  return q.submit([=](sycl::handler& h) {
    sycl::local_accessor<double, 0> srel_acc(h);
    sycl::local_accessor<int, 0> sfail_acc(h);
    sycl::local_accessor<int, 0> spiv_acc(h);
    sycl::local_accessor<int, 0> snext_acc(h);
    sycl::local_accessor<double, 1> tile_acc(
        sycl::range<1>(kTileBytes<SolverShapeFor<Ws>::sub> / 8), h);
    // One work-group: assembly, the panel LU, and the substitution.
    auto body = [=](sycl::nd_item<1> item) {
      double* tile = tile_acc.template get_multi_ptr<sycl::access::decorated::no>().get();
      double& srel = srel_acc;
      int& sfail = sfail_acc;
      int& spiv = spiv_acc;
      int& snext = snext_acc;
      solver_kernel<Ws>(item, tile, &srel, &sfail, &spiv, &snext, tab, row_off, row_reac, Tzone, gC, gC0,
                        gW, gRhs, gJ, gPerm, gKf, gK0, gLogKc, steps, dt, status, niters, gPart);
    };
    h.parallel_for<solver_kernel_name<Ws>>(ndr, SolverBody<Ws, decltype(body)>{body});
  });
}

// Sub-group widths the kernel is built for, in order of preference.
constexpr int kSubGroupWidths[] = {32, 64, 16};

// Launches the instantiation for sub-group width ws (one of kSubGroupWidths).
inline sycl::event submit_solver(sycl::queue& q, int ws, int zones, int blk, Tables tab,
                                 const int* row_off, const int* row_reac, const double* Tzone,
                                 double* gC, double* gC0, double* gW, double* gRhs, double* gJ,
                                 int* gPerm, double* gKf, double* gK0, double* gLogKc, int steps,
                                 double dt, int* status, int* niters, double* gPart) {
  switch (ws) {
    case 64:
      return submit_solver_ws<64>(q, zones, blk, tab, row_off, row_reac, Tzone, gC, gC0, gW, gRhs,
                                  gJ, gPerm, gKf, gK0, gLogKc, steps, dt, status, niters, gPart);
    case 32:
      return submit_solver_ws<32>(q, zones, blk, tab, row_off, row_reac, Tzone, gC, gC0, gW, gRhs,
                                  gJ, gPerm, gKf, gK0, gLogKc, steps, dt, status, niters, gPart);
    default:
      return submit_solver_ws<16>(q, zones, blk, tab, row_off, row_reac, Tzone, gC, gC0, gW, gRhs,
                                  gJ, gPerm, gKf, gK0, gLogKc, steps, dt, status, niters, gPart);
  }
}

// Host view of SolverShapeFor<ws>.
struct SolverShape {
  int block;
  int panel;
  int sub;
  int tile_bytes;
};

// Shape for a compile-time sub-group width.
template <int Ws>
SolverShape solver_shape_ws() {
  using S = SolverShapeFor<Ws>;
  return {S::block, S::panel, S::sub, kTileBytes<S::sub>};
}

// Shape for the device sub-group width.
inline SolverShape solver_shape(int ws) {
  switch (ws) {
    case 64: return solver_shape_ws<64>();
    case 32: return solver_shape_ws<32>();
    default: return solver_shape_ws<16>();
  }
}
