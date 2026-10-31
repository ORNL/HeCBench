#pragma once

// Host workload shared by the CUDA, HIP, and SYCL drivers. One copy, so the
// input draw and the acceptance comparison cannot drift between ports.

#include "reference.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

constexpr int kDim = 128;
constexpr int kFrames = 101;
constexpr int kHeads = 7;
constexpr int kDefaultTokens = 1008;
constexpr int kWarmup = 10;
constexpr double kRelTol = 1e-6;


// xorshift32 plus Box-Muller. The draw matches the reference script's
// statistics (unit-norm SiLU keys, SiLU values, sigmoid beta), not its bits.
struct Rng {
  uint32_t s;
  explicit Rng(uint32_t seed) : s(seed ? seed : 1u) {}
  uint32_t next() {
    uint32_t x = s;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    s = x;
    return x;
  }
  float u01() {
    return (next() >> 8) * (1.0f / 16777216.0f) + 1.0e-7f;
  }
  float normal() {
    const float u1 = u01();
    const float u2 = u01();
    return sqrtf(-2.0f * logf(u1)) * cosf(6.28318530718f * u2);
  }
};

float sigmoid(float x) {
  if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
  const float e = expf(x);
  return e / (1.0f + e);
}

float silu(float x) { return x * sigmoid(x); }

float softplus(float x) {
  if (x > 20.0f) return x;
  if (x < -20.0f) return expf(x);
  return log1pf(expf(x));
}

// A = K^T diag(beta) K and B = V^T diag(beta) K, then A is symmetrized.
// beta = sigmoid(.) * beta_scale. At scale 1, beta is in (0, 1), the model
// range. Scales 50 and 100 multiply that beta, so A and B grow by 50 and 100
// and alpha does not. The seed ignores beta_scale, so `make run` is one draw
// at three magnitudes. alpha = exp(-exp(a_log) * softplus(delta)), except
// frame 0, which is the virtual text frame and is fixed at 1.
void generate_inputs(int nmat, int heads, int tokens, float beta_scale,
                     std::vector<float>& A, std::vector<float>& B,
                     std::vector<float>& alpha) {
  const int d = kDim;
  std::vector<float> a_log(heads);
  Rng head_rng(1u);
  for (int h = 0; h < heads; ++h) a_log[h] = head_rng.normal();

  parallel_for(nmat, [&](int index) {
    const int frame = index / heads;
    const int head = index % heads;
    Rng rng(0x9E3779B9u ^ static_cast<uint32_t>(index) * 0x85EBCA6Bu);
    std::vector<float> K(static_cast<size_t>(tokens) * d);
    std::vector<float> V(static_cast<size_t>(tokens) * d);
    std::vector<float> beta(tokens);
    for (int t = 0; t < tokens; ++t) {
      float ss = 0.0f;
      for (int j = 0; j < d; ++j) {
        const float x = silu(rng.normal());
        K[static_cast<size_t>(t) * d + j] = x;
        ss += x * x;
      }
      const float inv = 1.0f / sqrtf(ss);
      for (int j = 0; j < d; ++j) K[static_cast<size_t>(t) * d + j] *= inv;
      for (int j = 0; j < d; ++j)
        V[static_cast<size_t>(t) * d + j] = silu(rng.normal());
      beta[t] = sigmoid(rng.normal()) * beta_scale;
    }
    float* arow = alpha.data() + static_cast<size_t>(index) * d;
    for (int j = 0; j < d; ++j) {
      if (frame == 0) {
        arow[j] = 1.0f;
      } else {
        arow[j] = expf(-expf(a_log[head]) * softplus(rng.normal()));
      }
    }
    float* Am = A.data() + static_cast<size_t>(index) * d * d;
    float* Bm = B.data() + static_cast<size_t>(index) * d * d;
    std::fill(Am, Am + d * d, 0.0f);
    std::fill(Bm, Bm + d * d, 0.0f);
    for (int t = 0; t < tokens; ++t) {
      const float bt = beta[t];
      const float* kt = K.data() + static_cast<size_t>(t) * d;
      const float* vt = V.data() + static_cast<size_t>(t) * d;
      for (int i = 0; i < d; ++i) {
        const float ki = bt * kt[i];
        const float vi = bt * vt[i];
        float* Ai = Am + i * d;
        float* Bi = Bm + i * d;
        for (int j = 0; j < d; ++j) {
          Ai[j] += ki * kt[j];
          Bi[j] += vi * kt[j];
        }
      }
    }
    for (int i = 0; i < d; ++i) {
      for (int j = 0; j < i; ++j) {
        const float s = 0.5f * (Am[i * d + j] + Am[j * d + i]);
        Am[i * d + j] = s;
        Am[j * d + i] = s;
      }
    }
  });
}

// Row of packed index t = r(r+1)/2 + c. The kernel no longer stores a packed
// triangle; the n=8 self-check still inverts this map.
int tri_row(int t) {
  int r = static_cast<int>((-1.0 + sqrt(1.0 + 8.0 * static_cast<double>(t))) * 0.5);
  if (r * (r + 1) / 2 > t) --r;
  if ((r + 1) * (r + 2) / 2 <= t) ++r;
  return r;
}


struct ErrorStats {
  double rel_t;
  double rel_j;
  double max_t;
  double max_j;
  double asym;
  float worst_ar, worst_ac, worst_trc, worst_tcr;
  bool finite;
};

// Frobenius relative error of T and J against the fp64 inverse, plus the
// largest absolute entrywise gap. asym is the largest absolute gap between
// T[r,c]/alpha[r] and T[c,r]/alpha[c]. Rows with alpha < 1e-20 are skipped:
// alpha underflows to ~1e-45 there, and alpha * inv is zero in fp32, so the
// quotient is not a meaningful symmetry check. The pass bar for asym is 1e-4.
ErrorStats compare(const float* T, const float* J, const double* T64,
                   const double* J64, const float* alpha, size_t nmat) {
  ErrorStats s{};
  s.finite = true;
  double nt = 0, nj = 0, dt = 0, dj = 0;
  const size_t nn = static_cast<size_t>(kDim) * kDim;
  for (size_t m = 0; m < nmat; ++m) {
    for (int i = 0; i < kDim * kDim; ++i) {
      const float tv = T[m * nn + i];
      const float jv = J[m * nn + i];
      if (!std::isfinite(tv) || !std::isfinite(jv)) s.finite = false;
      const double td = static_cast<double>(tv) - T64[m * nn + i];
      const double jd = static_cast<double>(jv) - J64[m * nn + i];
      dt += td * td;
      dj += jd * jd;
      nt += T64[m * nn + i] * T64[m * nn + i];
      nj += J64[m * nn + i] * J64[m * nn + i];
      s.max_t = std::max(s.max_t, std::fabs(td));
      s.max_j = std::max(s.max_j, std::fabs(jd));
    }
    const float* a = alpha + m * kDim;
    const float* Tm = T + m * nn;
    for (int r = 0; r < kDim; ++r) {
      for (int c = r + 1; c < kDim; ++c) {
        if (a[r] < 1e-20f || a[c] < 1e-20f) continue;
        const double ir = static_cast<double>(Tm[r * kDim + c]) / a[r];
        const double ic = static_cast<double>(Tm[c * kDim + r]) / a[c];
        const double rel = std::fabs(ir - ic);
        if (rel > s.asym) {
          s.asym = rel;
          s.worst_ar = a[r];
          s.worst_ac = a[c];
          s.worst_trc = Tm[r * kDim + c];
          s.worst_tcr = Tm[c * kDim + r];
        }
      }
    }
  }
  s.rel_t = std::sqrt(dt / nt);
  s.rel_j = std::sqrt(dj / nj);
  return s;
}

bool self_check_reference() {
  bool tri_ok = true;
  for (int r = 0; r < kDim && tri_ok; ++r) {
    for (int c = 0; c <= r; ++c) {
      const int t = r * (r + 1) / 2 + c;
      const int rr = tri_row(t);
      if (rr != r || t - rr * (rr + 1) / 2 != c) tri_ok = false;
    }
  }

  constexpr int n = 8;
  std::vector<double> A(n * n, 0.0), B(n * n), alpha(n), M(n * n);
  Rng rng(42u);
  for (int t = 0; t < 16; ++t) {
    double x[n], y[n];
    for (int i = 0; i < n; ++i) {
      x[i] = rng.normal();
      y[i] = rng.normal();
    }
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        A[i * n + j] += x[i] * x[j];
        B[i * n + j] += y[i] * x[j];
      }
  }
  for (int i = 0; i < n; ++i) {
    alpha[i] = 0.25 + 0.05 * i;
    for (int j = 0; j < n; ++j) M[i * n + j] = A[i * n + j];
    M[i * n + i] += 1.0;
  }
  std::vector<double> T(n * n), J(n * n), inv;
  today_reference(A.data(), B.data(), alpha.data(), T.data(), J.data(), n);
  const bool inverted = invert_gauss(M.data(), inv, n);
  double max_inv = 0, max_j = 0;
  if (inverted) {
    for (int r = 0; r < n; ++r) {
      for (int c = 0; c < n; ++c) {
        max_inv = std::max(max_inv, std::fabs(T[r * n + c] / alpha[r] - inv[r * n + c]));
        double js = 0;
        for (int k = 0; k < n; ++k) js += B[r * n + k] * inv[k * n + c];
        max_j = std::max(max_j, std::fabs(J[r * n + c] - js));
      }
    }
  }
  const bool ok = tri_ok && inverted && max_inv < 1e-8 && max_j < 1e-8;
  printf("four-call reference self-check (n=8 vs fp64 inv): max |inv| %.3e max |J| %.3e %s\n",
         max_inv, max_j, ok ? "OK" : "FAIL");
  return ok;
}
