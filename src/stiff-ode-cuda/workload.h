// Element-balanced isomer network with the species and reaction counts of the
// detailed n-heptane mechanism (mechanism.h). Its chemistry is not meant to be
// representative. Every reaction is a first-order edge inside one of 11
// element families, so the analytic Jacobian is sparse. The solver still
// stores that matrix and factors it with a dense 560x560 LU.
#pragma once

#include "mechanism.h"

#include <vector>

inline constexpr int kSynthBase = 11;

// O, H, C, N, Ar for H2, H, O, O2, OH, H2O, CH4, CO, CO2, N2, Ar.
inline constexpr int kSynthComp[kSynthBase * NE] = {
    0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 2, 0, 0, 0,
    0, 4, 1, 0, 0, 1, 0, 1, 0, 0, 2, 0, 1, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1,
};

struct SynthIds {
  int h2 = 0;
  int h = 1;
  int o = 2;
  int o2 = 3;
  int oh = 4;
  int ch4 = 6;
  int n2 = 9;
};

struct SynthMech {
  std::vector<int> rtype, reversible, nreact, nprod;
  std::vector<int> r_idx, r_nu, p_idx, p_nu;
  std::vector<int> eff_off, eff_len, eff_sp, comp;
  std::vector<double> eff_eps, troe;
  std::vector<double> A_high, B_high, Ea_high, A_low, B_low, Ea_low;
  std::vector<double> nasa_lo, nasa_hi, tmid;
  SynthIds ids;
};

// Builds the element-balanced isomer table with NS species and NR reactions.
inline void build_synthetic(SynthMech& m) {
  m = SynthMech{};
  m.comp.assign((size_t)NS * NE, 0);
  for (int s = 0; s < NS; ++s) {
    int b = s % kSynthBase;
    for (int e = 0; e < NE; ++e) m.comp[(size_t)s * NE + e] = kSynthComp[b * NE + e];
  }
  m.rtype.assign(NR, 0);
  m.reversible.assign(NR, 1);
  m.nreact.assign(NR, 1);
  m.nprod.assign(NR, 1);
  m.r_idx.assign((size_t)NR * 3, 0);
  m.r_nu.assign((size_t)NR * 3, 0);
  m.p_idx.assign((size_t)NR * 3, 0);
  m.p_nu.assign((size_t)NR * 3, 0);
  m.eff_off.assign(NR, 0);
  m.eff_len.assign(NR, 0);
  m.eff_sp.assign(N_EFF, 0);
  m.eff_eps.assign(N_EFF, 1.0);
  m.troe.assign((size_t)NR * 4, 0.0);
  m.A_high.assign(NR, 0.0);
  m.B_high.assign(NR, 0.0);
  m.Ea_high.assign(NR, 0.0);
  m.A_low.assign(NR, 0.0);
  m.B_low.assign(NR, 0.0);
  m.Ea_low.assign(NR, 0.0);
  m.nasa_lo.assign((size_t)NS * 7, 0.0);
  m.nasa_hi.assign((size_t)NS * 7, 0.0);
  m.tmid.assign(NS, 1000.0);
  for (int s = 0; s < NS; ++s) {
    m.nasa_lo[(size_t)s * 7] = 3.5;
    m.nasa_lo[(size_t)s * 7 + 5] = -1.0e3;
    m.nasa_hi[(size_t)s * 7] = 3.5;
    m.nasa_hi[(size_t)s * 7 + 5] = -1.0e3;
  }
  const int slots = (NS - 1) / kSynthBase;
  for (int r = 0; r < NR; ++r) {
    int base = r % kSynthBase;
    int slot = 1 + (r / kSynthBase) % slots;
    int prod = base + slot * kSynthBase;
    if (prod >= NS) prod = base + kSynthBase;
    m.r_idx[(size_t)r * 3] = base;
    m.r_nu[(size_t)r * 3] = 1;
    m.p_idx[(size_t)r * 3] = prod;
    m.p_nu[(size_t)r * 3] = 1;
    m.A_high[r] = 1.0e7 * (1.0 + (r % 5));
    m.Ea_high[r] = 8000.0 + 250.0 * (r % 17);
  }
}
