// Derived from Laghos' partial-assembly force operator.
//
// Copyright (c) 2017 Lawrence Livermore National Security, LLC.
// SPDX-License-Identifier: BSD-2-Clause

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace laghos_force {

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LAGHOS_HD __host__ __device__
#else
#define LAGHOS_HD
#endif

constexpr int DIM = 3;
constexpr double kEpsSquared = 4.9303806576313237838e-32; // epsilon(double)^2

// Specialized 3D Laghos sizes: Q1-Q0, Q2-Q1, Q3-Q2, and Q4-Q3.
inline bool supported_points(int points) {
  return points == 8 || points == 64 || points == 216 || points == 512;
}

inline int default_elements(int points) {
  switch (points) {
  case 8:
    return 524288; // Q1-Q0
  case 64:
    return 65536; // Q2-Q1, ATS priority 1
  case 216:
    return 19418; // Q3-Q2, ATS priority 2
  case 512:
    return 8192; // Q4-Q3, same total sample count as priority 1
  default:
    return 0;
  }
}

LAGHOS_HD inline std::size_t stress_index(int q, int e, int i, int c, int ne,
                                          int qqq) {
  return q + static_cast<std::size_t>(qqq) *
                 (e + static_cast<std::size_t>(ne) * (i + DIM * c));
}

LAGHOS_HD inline std::size_t output_index(int d, int c, int e, int ddd) {
  return d + static_cast<std::size_t>(ddd) * (c + DIM * e);
}

inline double flops_per_element(int d1d, int q1d, int l1d) {
  const double d = d1d;
  const double q = q1d;
  const double l = l1d;
  const double q3 = q * q * q;
  return 4.0 * q3 * l * l * l + 27.0 * d * q3 + 18.0 * d * d * q * q +
         18.0 * d * d * d * q;
}

inline double bernstein(int degree, int index, double x) {
  const double t = 0.5 * (x + 1.0);
  const double u = 1.0 - t;
  double value = 1.0;
  for (int i = 0; i < index; ++i)
    value *= t;
  for (int i = 0; i < degree - index; ++i)
    value *= u;
  for (int i = 1; i <= index; ++i)
    value *= static_cast<double>(degree - index + i) / static_cast<double>(i);
  return value;
}

inline void gauss_nodes(int q1d, double *nodes) {
  switch (q1d) {
  case 2:
    nodes[0] = -0.57735026918962576451;
    nodes[1] = 0.57735026918962576451;
    break;
  case 4:
    nodes[0] = -0.86113631159405257522;
    nodes[1] = -0.33998104358485626480;
    nodes[2] = 0.33998104358485626480;
    nodes[3] = 0.86113631159405257522;
    break;
  case 6:
    nodes[0] = -0.93246951420315202781;
    nodes[1] = -0.66120938646626451366;
    nodes[2] = -0.23861918608319690863;
    nodes[3] = 0.23861918608319690863;
    nodes[4] = 0.66120938646626451366;
    nodes[5] = 0.93246951420315202781;
    break;
  case 8:
    nodes[0] = -0.96028985649753623168;
    nodes[1] = -0.79666647741362673959;
    nodes[2] = -0.52553240991632898582;
    nodes[3] = -0.18343464249564980494;
    nodes[4] = 0.18343464249564980494;
    nodes[5] = 0.52553240991632898582;
    nodes[6] = 0.79666647741362673959;
    nodes[7] = 0.96028985649753623168;
    break;
  default:
    break;
  }
}

inline void gll_nodes(int d1d, double *nodes) {
  switch (d1d) {
  case 2:
    nodes[0] = -1.0;
    nodes[1] = 1.0;
    break;
  case 3:
    nodes[0] = -1.0;
    nodes[1] = 0.0;
    nodes[2] = 1.0;
    break;
  case 4:
    nodes[0] = -1.0;
    nodes[1] = -0.44721359549995793928;
    nodes[2] = 0.44721359549995793928;
    nodes[3] = 1.0;
    break;
  case 5:
    nodes[0] = -1.0;
    nodes[1] = -0.65465367070797714380;
    nodes[2] = 0.0;
    nodes[3] = 0.65465367070797714380;
    nodes[4] = 1.0;
    break;
  default:
    break;
  }
}

inline double lagrange(int n, const double *nodes, int i, double x) {
  double value = 1.0;
  for (int j = 0; j < n; ++j) {
    if (j != i)
      value *= (x - nodes[j]) / (nodes[i] - nodes[j]);
  }
  return value;
}

inline double lagrange_derivative(int n, const double *nodes, int i, double x) {
  double derivative = 0.0;
  for (int m = 0; m < n; ++m) {
    if (m == i)
      continue;
    double term = 1.0 / (nodes[i] - nodes[m]);
    for (int j = 0; j < n; ++j) {
      if (j != i && j != m)
        term *= (x - nodes[j]) / (nodes[i] - nodes[j]);
    }
    derivative += term;
  }
  return derivative;
}

inline void make_basis(int d1d, int q1d, int l1d, double *l2b, double *h1bt,
                       double *h1gt) {
  double quadrature[8];
  double nodes[5];
  gauss_nodes(q1d, quadrature);
  gll_nodes(d1d, nodes);
  for (int q = 0; q < q1d; ++q) {
    for (int l = 0; l < l1d; ++l)
      l2b[q * l1d + l] = bernstein(l1d - 1, l, quadrature[q]);
    for (int d = 0; d < d1d; ++d) {
      h1bt[d * q1d + q] = lagrange(d1d, nodes, d, quadrature[q]);
      h1gt[d * q1d + q] = lagrange_derivative(d1d, nodes, d, quadrature[q]);
    }
  }
}

inline double sample(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return static_cast<double>(x >> 11) * 0x1.0p-53;
}

inline void initialize(int ne, int q1d, int l1d, std::vector<double> &energy,
                       std::vector<double> &stress) {
  const int qqq = q1d * q1d * q1d;
  const int lll = l1d * l1d * l1d;
  energy.resize(static_cast<std::size_t>(ne) * lll);
  stress.resize(static_cast<std::size_t>(ne) * qqq * DIM * DIM);
  for (int e = 0; e < ne; ++e) {
    for (int l = 0; l < lll; ++l) {
      energy[l + static_cast<std::size_t>(lll) * e] =
          0.5 + sample(17ULL * e + l);
    }
    for (int c = 0; c < DIM; ++c) {
      for (int i = 0; i < DIM; ++i) {
        for (int q = 0; q < qqq; ++q) {
          const double perturb =
              0.05 * (sample(1009ULL * e + 97ULL * q + 11ULL * i + c) - 0.5);
          stress[stress_index(q, e, i, c, ne, qqq)] =
              (i == c ? 1.0 : 0.15 * (i + c + 1)) + perturb;
        }
      }
    }
  }
}

inline void reference_element(int e, int ne, int d1d, int q1d, int l1d,
                              const double *l2b, const double *h1bt,
                              const double *h1gt, const double *stress,
                              const double *energy, double *velocity) {
  const int qqq = q1d * q1d * q1d;
  const int lll = l1d * l1d * l1d;
  const int ddd = d1d * d1d * d1d;
  std::vector<double> eq(qqq);
  for (int qz = 0; qz < q1d; ++qz) {
    for (int qy = 0; qy < q1d; ++qy) {
      for (int qx = 0; qx < q1d; ++qx) {
        double v = 0.0;
        for (int lz = 0; lz < l1d; ++lz)
          for (int ly = 0; ly < l1d; ++ly)
            for (int lx = 0; lx < l1d; ++lx) {
              const int l = lx + l1d * (ly + l1d * lz);
              v += l2b[qx * l1d + lx] * l2b[qy * l1d + ly] *
                   l2b[qz * l1d + lz] *
                   energy[l + static_cast<std::size_t>(lll) * e];
            }
        eq[qx + q1d * (qy + q1d * qz)] = v;
      }
    }
  }

  for (int c = 0; c < DIM; ++c)
    for (int hz = 0; hz < d1d; ++hz)
      for (int hy = 0; hy < d1d; ++hy)
        for (int hx = 0; hx < d1d; ++hx) {
          double v = 0.0;
          for (int qz = 0; qz < q1d; ++qz)
            for (int qy = 0; qy < q1d; ++qy)
              for (int qx = 0; qx < q1d; ++qx) {
                const int q = qx + q1d * (qy + q1d * qz);
                const double sx = stress[stress_index(q, e, 0, c, ne, qqq)];
                const double sy = stress[stress_index(q, e, 1, c, ne, qqq)];
                const double sz = stress[stress_index(q, e, 2, c, ne, qqq)];
                v += eq[q] * (h1gt[hx * q1d + qx] * h1bt[hy * q1d + qy] *
                                  h1bt[hz * q1d + qz] * sx +
                              h1bt[hx * q1d + qx] * h1gt[hy * q1d + qy] *
                                  h1bt[hz * q1d + qz] * sy +
                              h1bt[hx * q1d + qx] * h1bt[hy * q1d + qy] *
                                  h1gt[hz * q1d + qz] * sz);
              }
          const int d = hx + d1d * (hy + d1d * hz);
          velocity[output_index(d, c, e, ddd)] =
              (v > -kEpsSquared && v < kEpsSquared) ? 0.0 : v;
        }
}

inline double max_relative_error(int ne, int checked, int d1d, int q1d, int l1d,
                                 const double *l2b, const double *h1bt,
                                 const double *h1gt, const double *stress,
                                 const double *energy, const double *velocity) {
  const int ddd = d1d * d1d * d1d;
  std::vector<double> ref(static_cast<std::size_t>(checked) * DIM * ddd);
  double error = 0.0;
  for (int e = 0; e < checked; ++e) {
    reference_element(e, ne, d1d, q1d, l1d, l2b, h1bt, h1gt, stress, energy,
                      ref.data());
    for (int c = 0; c < DIM; ++c)
      for (int d = 0; d < ddd; ++d) {
        const std::size_t got_i = output_index(d, c, e, ddd);
        const std::size_t ref_i = output_index(d, c, e, ddd);
        const double scale = std::max(1.0, std::abs(ref[ref_i]));
        const double err = std::abs(velocity[got_i] - ref[ref_i]) / scale;
        if (!std::isfinite(err))
          return err;
        error = std::max(error, err);
      }
  }
  return error;
}

} // namespace laghos_force

#undef LAGHOS_HD
