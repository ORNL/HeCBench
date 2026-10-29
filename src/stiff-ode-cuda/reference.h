// Host reference for the stiff step, shared by the CUDA, HIP, SYCL, and OpenMP
// ports. Include this after the backend kinetics header. assemble_row,
// prepare_one, and apply_newton come from there. host_reference spreads the
// zones over OpenMP threads when the driver is built with OpenMP; it uses only
// directives, so it also builds (serially) without it.
#pragma once

#include <algorithm>
#include <cmath>
#include <vector>

struct StepResult {
  int code;
  double rel;
  int iters;
};

// CH4/air state at a zone-dependent temperature and equivalence ratio.
template <int SpeciesCount>
void init_zone(int zone, double& T, double* C, int id_ch4, int id_o2, int id_n2, int id_h, int id_o,
               int id_oh, int id_h2) {
  T = 1400.0 + 50.0 * (zone % 9);
  double phi = 0.8 + 0.05 * (zone % 5);
  double n_ch4 = 1.0;
  double n_o2 = 2.0 / phi;
  double n_n2 = 3.76 * n_o2;
  double ntot = n_ch4 + n_o2 + n_n2;
  double ctot = kPstd / (kRerg * T);
  for (int i = 0; i < SpeciesCount; ++i) C[i] = 0.0;
  C[id_ch4] = n_ch4 / ntot * ctot;
  C[id_o2] = n_o2 / ntot * ctot;
  C[id_n2] = n_n2 / ntot * ctot;
  double seed = 1e-6 * ctot;
  C[id_h] += seed;
  C[id_o] += seed;
  C[id_oh] += seed;
  C[id_h2] += seed;
}

// Analytic Jacobian and production rates for every species row.
template <int SpeciesCount>
void assemble_all(const Tables& tab, const double* C, const double* kf, const double* k0,
                  const double* logKc, double T, const int* row_off, const int* row_reac,
                  double* J, double* w) {
  for (int i = 0; i < SpeciesCount; ++i)
    assemble_row<SpeciesCount>(tab, i, C, kf, k0, logKc, T, row_off, row_reac, J, w);
}

// Production rates only. Leaves a stored factorization of J in place.
template <int SpeciesCount>
void assemble_rates(const Tables& tab, const double* C, const double* kf, const double* k0,
                    const double* logKc, double T, const int* row_off, const int* row_reac,
                    double* J, double* w) {
  for (int i = 0; i < SpeciesCount; ++i)
    assemble_row<SpeciesCount, false>(tab, i, C, kf, k0, logKc, T, row_off, row_reac, J, w);
}

// Arrhenius rates and log(Kc) for every reaction at temperature T.
template <int ReactionCount>
void prepare_all(const Tables& tab, double T, double* kf, double* k0, double* logKc) {
  for (int r = 0; r < ReactionCount; ++r) prepare_one(tab, r, T, kf[r], k0[r], logKc[r]);
}

// Modified Newton, matching the device: J is assembled and factored on the
// first iteration of the step, and later iterations reassemble only ω.
template <int SpeciesCount, int ReactionCount>
StepResult host_step(const Tables& tab, const double* C0_in, double* C, double* C0, double* w,
                     double* J, double* rhs, int* perm, const double* kf, const double* k0,
                     const double* logKc, double T, double dt, const int* row_off,
                     const int* row_reac) {
  for (int i = 0; i < SpeciesCount; ++i) C0[i] = C0_in ? C0_in[i] : C[i];
  if (C0_in) {
    for (int i = 0; i < SpeciesCount; ++i) C[i] = C0_in[i];
  }
  StepResult s{2, 1.0, kNewtonMax};
  for (int it = 0; it < kNewtonMax; ++it) {
    const bool factor = it == 0;
    if (factor)
      assemble_all<SpeciesCount>(tab, C, kf, k0, logKc, T, row_off, row_reac, J, w);
    else
      assemble_rates<SpeciesCount>(tab, C, kf, k0, logKc, T, row_off, row_reac, J, w);
    double rel = apply_newton<SpeciesCount>(C, C0, w, J, rhs, perm, dt, factor);
    s.rel = rel;
    s.iters = it + 1;
    if (!(rel >= 0.0)) {
      s.code = 1;
      return s;
    }
    if (rel < kNewtonTol) {
      s.code = 0;
      return s;
    }
  }
  s.code = 2;
  return s;
}

// Integrates every zone of C (zone-major, SpeciesCount per zone) for steps
// backward-Euler steps, one zone per OpenMP thread. Returns the number of
// zones whose Newton iteration failed.
template <int SpeciesCount, int ReactionCount>
int host_reference(const Tables& tab, const std::vector<double>& temps, std::vector<double>& C,
                   int steps, double dt, const int* row_off, const int* row_reac) {
  const int zones = (int)temps.size();
  int bad = 0;
#pragma omp parallel reduction(+ : bad)
  {
    std::vector<double> kf(ReactionCount), k0(ReactionCount), logKc(ReactionCount);
    std::vector<double> C0(SpeciesCount), w(SpeciesCount), rhs(SpeciesCount);
    std::vector<double> J((size_t)SpeciesCount * SpeciesCount);
    std::vector<int> perm(SpeciesCount);
#pragma omp for schedule(dynamic)
    for (int z = 0; z < zones; ++z) {
      double* Cz = C.data() + (size_t)z * SpeciesCount;
      prepare_all<ReactionCount>(tab, temps[z], kf.data(), k0.data(), logKc.data());
      for (int s = 0; s < steps; ++s) {
        StepResult st = host_step<SpeciesCount, ReactionCount>(
            tab, nullptr, Cz, C0.data(), w.data(), J.data(), rhs.data(), perm.data(), kf.data(),
            k0.data(), logKc.data(), temps[z], dt, row_off, row_reac);
        if (st.code != 0) {
          ++bad;
          break;
        }
      }
    }
  }
  return bad;
}

// Largest element imbalance in the production rates, relative to their scale.
template <int SpeciesCount, int ElementCount>
double element_residual(const double* w, const int* comp) {
  double elem[ElementCount] = {};
  double scale = 0.0;
  for (int i = 0; i < SpeciesCount; ++i) {
    scale = std::max(scale, std::fabs(w[i]));
    for (int e = 0; e < ElementCount; ++e) elem[e] += (double)comp[i * ElementCount + e] * w[i];
  }
  double err = 0.0;
  for (int e = 0; e < ElementCount; ++e) err = std::max(err, std::fabs(elem[e]));
  return err / std::max(scale, 1e-30);
}

// Largest change in elemental abundance from C0 to C1, relative to C0.
template <int SpeciesCount, int ElementCount>
double element_drift(const double* C0, const double* C1, const int* comp) {
  double a[ElementCount] = {};
  double b[ElementCount] = {};
  double scale = 0.0;
  for (int i = 0; i < SpeciesCount; ++i) {
    for (int e = 0; e < ElementCount; ++e) {
      double n = (double)comp[i * ElementCount + e];
      a[e] += n * C0[i];
      b[e] += n * C1[i];
    }
    scale = std::max(scale, std::fabs(C0[i]));
  }
  double err = 0.0;
  for (int e = 0; e < ElementCount; ++e) err = std::max(err, std::fabs(b[e] - a[e]));
  double enorm = 0.0;
  for (int e = 0; e < ElementCount; ++e) enorm = std::max(enorm, std::fabs(a[e]));
  return err / std::max(enorm, 1e-30);
}

// Largest relative gap between the analytic Jacobian and a centered difference.
template <int SpeciesCount, int ReactionCount>
double jacobian_fd_error(const Tables& tab, double* C, const double* kf, const double* k0,
                         const double* logKc, double T, const int* row_off, const int* row_reac,
                         const double* Jexact) {
  std::vector<double> saved(C, C + SpeciesCount);
  std::vector<double> J(SpeciesCount * SpeciesCount, 0.0);
  std::vector<double> w(SpeciesCount, 0.0);
  std::vector<double> wp(SpeciesCount, 0.0);
  std::vector<double> wm(SpeciesCount, 0.0);
  double maxj = 0.0;
  for (int i = 0; i < SpeciesCount * SpeciesCount; ++i) maxj = std::max(maxj, std::fabs(Jexact[i]));
  double worst = 0.0;
  const int ncols = SpeciesCount <= 64 ? SpeciesCount : 4;
  for (int k = 0; k < ncols; ++k) {
    double h = std::max(1e-7 * std::fabs(saved[k]), 1e-14);
    for (int i = 0; i < SpeciesCount; ++i) C[i] = saved[i];
    C[k] = saved[k] + h;
    assemble_all<SpeciesCount>(tab, C, kf, k0, logKc, T, row_off, row_reac, J.data(), wp.data());
    for (int i = 0; i < SpeciesCount; ++i) C[i] = saved[i];
    C[k] = saved[k] - h;
    assemble_all<SpeciesCount>(tab, C, kf, k0, logKc, T, row_off, row_reac, J.data(), wm.data());
    for (int i = 0; i < SpeciesCount; ++i) {
      double fd = (wp[i] - wm[i]) / (2.0 * h);
      double an = Jexact[i * SpeciesCount + k];
      double mag = std::max(std::fabs(an), std::fabs(fd));
      if (mag < 1e-8 * std::max(maxj, 1.0)) continue;
      worst = std::max(worst, std::fabs(an - fd) / mag);
    }
  }
  for (int i = 0; i < SpeciesCount; ++i) C[i] = saved[i];
  return worst;
}

struct Rel {
  double max_rel = 0.0;
  bool finite = true;
};

// Largest relative difference between two buffers.
inline Rel compare_buf(const std::vector<double>& a, const std::vector<double>& b) {
  Rel r;
  size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i) {
    if (!std::isfinite(a[i]) || !std::isfinite(b[i])) {
      r.finite = false;
      continue;
    }
    double diff = std::fabs(a[i] - b[i]);
    double den = std::max(std::fabs(a[i]), std::fabs(b[i]));
    double rel = den == 0.0 ? (diff == 0.0 ? 0.0 : 1.0) : diff / den;
    if (rel > r.max_rel) r.max_rel = rel;
  }
  return r;
}
