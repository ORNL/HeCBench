// The portUrb WENO finite-volume dynamical-core tendency benchmark.
//
// The compute_tendencies routine performs, for a collocated finite-volume
// A-grid:
//   1. acoustic upwind reconstruction of edge pressure & normal momentum in x,
//   y, z
//   2. advective upwind reconstruction of edge quantities and total fluxes in
//   x, y, z
//   3. flux-divergence + gravity source to form the state tendency
// The reconstruction uses 9th-order WENO limiting, which is the dominant cost.
//
// This benchmark targets portUrb's obstacle-resolving (urban/turbine) use case,
// so it includes immersed-boundary handling: an immersed-proportion field, a
// data-dependent stencil rewrite (modify_stencil_immersed_der0), and the WENO
// immersed-edge (immL/immR) smoothness modification. These add extra halo loads
// and divergent branching representative of real runs.
//
// Framework-only aspects that don't change the compute character on a uniform
// single-GPU grid are dropped: MPI halo exchange, solid-wall BCs, Coriolis, and
// the vertical metric terms of a stretched grid (a uniform vertical grid makes
// those factors identically one).
//

#ifndef WENOFV_KERNELS_H
#define WENOFV_KERNELS_H

#include <cmath>
#include <cstddef>

#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif

static constexpr int ORD = 9;            // WENO order of accuracy
static constexpr int HS = (ORD + 1) / 2; // halo size (=5)
static constexpr int HSM1 = HS - 1;      // (=4) matches portUrb's "hsm1"
static constexpr int NUM_STATE = 5;      // rho, rho*u, rho*v, rho*w, rho*theta

// variable indices inside the padded working array "fields"
static constexpr int idR = 0; // density (perturbation from hydrostasis)
static constexpr int idU = 1; // u   (specific, i.e. divided by density)
static constexpr int idV = 2; // v
static constexpr int idW = 3; // w
static constexpr int idT = 4; // potential temperature (perturbation)
static constexpr int idP = 5; // pressure (perturbation)
static constexpr int NV = 6;  // NUM_STATE + 1 (pressure)

static constexpr float IMM_TH = 0.5f; // immersed-proportion threshold

// Grid + physical parameters passed by value to every kernel.
struct Grid {
  int nx, ny, nz;   // interior cell counts
  int px, py, pz;   // padded counts (n + 2*HS)
  float cs;         // speed of sound
  float grav;       // gravity
  float r_dx, r_dy; // 1/dx, 1/dy
  float r_dz;       // 1/dz (uniform vertical grid)
};

// (C-style, rightmost fastest)
__host__ __device__ inline size_t fidx(int v, int k, int j, int i,
                                       const Grid &g) {
  return (((size_t)v * g.pz + k) * g.py + j) * g.px +
         i; // fields: (NV, pz, py, px)
}
__host__ __device__ inline size_t pidx(int k, int j, int i, const Grid &g) {
  return ((size_t)k * g.py + j) * g.px + i; // immersed_prop: (pz, py, px)
}
__host__ __device__ inline size_t x3(int k, int j, int i, int Y, int X) {
  return ((size_t)k * Y + j) * X + i;
}
__host__ __device__ inline size_t x4(int v, int k, int j, int i, int Z, int Y,
                                     int X) {
  return (((size_t)v * Z + k) * Y + j) * X + i;
}

// WENO limiter Ported verbatim (float) from portUrb
// WenoLimiter<real,9>::value_based, including the immersed-edge smoothness
// modification (immL/immR). Returns left/right edge values L,R.
//
// Templated on which edges are actually consumed: the smoothness indicators
// (TV0..TV4) are shared, but the left- and right-edge reconstructions are
// independent, so callers that need only one edge (all acoustic
// reconstructions) skip the other. The arithmetic of every edge that IS
// computed is unchanged, so results stay bit-identical.
template <bool NeedL, bool NeedR>
__host__ __device__ inline void weno9_impl(const float v[ORD], float &L,
                                           float &R, bool immL, bool immR) {
  float TV0 =
      4.4956349206349206f * v[0] * v[0] - 41.369246031746032f * v[0] * v[1] +
      72.393452380952381f * v[0] * v[2] - 57.144246031746032f * v[0] * v[3] +
      17.128769841269841f * v[0] * v[4] + 95.825992063492063f * v[1] * v[1] -
      338.17380952380952f * v[1] * v[2] + 269.53531746031746f * v[1] * v[3] -
      81.644246031746032f * v[1] * v[4] + 301.86369047619048f * v[2] * v[2] -
      488.50714285714286f * v[2] * v[3] + 150.56011904761905f * v[2] * v[4] +
      202.49265873015873f * v[3] * v[3] - 128.86924603174603f * v[3] * v[4] +
      21.412301587301587f * v[4] * v[4];
  float TV1 =
      1.3706349206349206f * v[1] * v[1] - 12.077579365079365f * v[1] * v[2] +
      19.685119047619048f * v[1] * v[3] - 13.935912698412698f * v[1] * v[4] +
      3.5871031746031746f * v[1] * v[5] + 27.492658730158730f * v[2] * v[2] -
      92.257142857142857f * v[2] * v[3] + 66.868650793650794f * v[2] * v[4] -
      17.519246031746032f * v[2] * v[5] + 80.613690476190476f * v[3] * v[3] -
      121.42380952380952f * v[3] * v[4] + 32.768452380952381f * v[3] * v[5] +
      48.159325396825397f * v[4] * v[4] - 27.827579365079365f * v[4] * v[5] +
      4.4956349206349206f * v[5] * v[5];
  float TV2 =
      1.3706349206349206f * v[2] * v[2] - 10.119246031746032f * v[2] * v[3] +
      13.476785714285714f * v[2] * v[4] - 7.7275793650793651f * v[2] * v[5] +
      1.6287698412698413f * v[2] * v[6] + 20.825992063492063f * v[3] * v[3] -
      59.340476190476190f * v[3] * v[4] + 35.535317460317460f * v[3] * v[5] -
      7.7275793650793651f * v[3] * v[6] + 45.863690476190476f * v[4] * v[4] -
      59.340476190476190f * v[4] * v[5] + 13.476785714285714f * v[4] * v[6] +
      20.825992063492063f * v[5] * v[5] - 10.119246031746032f * v[5] * v[6] +
      1.3706349206349206f * v[6] * v[6];
  float TV3 =
      4.4956349206349206f * v[3] * v[3] - 27.827579365079365f * v[3] * v[4] +
      32.768452380952381f * v[3] * v[5] - 17.519246031746032f * v[3] * v[6] +
      3.5871031746031746f * v[3] * v[7] + 48.159325396825397f * v[4] * v[4] -
      121.42380952380952f * v[4] * v[5] + 66.868650793650794f * v[4] * v[6] -
      13.935912698412698f * v[4] * v[7] + 80.613690476190476f * v[5] * v[5] -
      92.257142857142857f * v[5] * v[6] + 19.685119047619048f * v[5] * v[7] +
      27.492658730158730f * v[6] * v[6] - 12.077579365079365f * v[6] * v[7] +
      1.3706349206349206f * v[7] * v[7];
  float TV4 =
      21.412301587301587f * v[4] * v[4] - 128.86924603174603f * v[4] * v[5] +
      150.56011904761905f * v[4] * v[6] - 81.644246031746032f * v[4] * v[7] +
      17.128769841269841f * v[4] * v[8] + 202.49265873015873f * v[5] * v[5] -
      488.50714285714286f * v[5] * v[6] + 269.53531746031746f * v[5] * v[7] -
      57.144246031746032f * v[5] * v[8] + 301.86369047619048f * v[6] * v[6] -
      338.17380952380952f * v[6] * v[7] + 72.393452380952381f * v[6] * v[8] +
      95.825992063492063f * v[7] * v[7] - 41.369246031746032f * v[7] * v[8] +
      4.4956349206349206f * v[8] * v[8];
  TV0 = fabsf(TV0);
  TV1 = fabsf(TV1);
  TV2 = fabsf(TV2);
  TV3 = fabsf(TV3);
  TV4 = fabsf(TV4);
  float r_sm = 1.0f / fmaxf(1.e-20f, TV0 + TV1 + TV2 + TV3 + TV4);
  TV0 *= r_sm;
  TV1 *= r_sm;
  TV2 *= r_sm;
  TV3 *= r_sm;
  TV4 *= r_sm;
  if (TV0 == 0)
    TV0 = fminf(TV1, fminf(TV2, fminf(TV3, TV4)));
  if (TV4 == 0)
    TV4 = fminf(TV0, fminf(TV1, fminf(TV2, TV3)));
  if (immL)
    TV4 = fmaxf(TV0, fmaxf(TV1, fmaxf(TV2, fmaxf(TV3, TV4))));
  if (immR)
    TV0 = fmaxf(TV0, fmaxf(TV1, fmaxf(TV2, fmaxf(TV3, TV4))));
  TV0 *= TV0;
  TV1 *= TV1;
  TV2 *= TV2;
  TV3 *= TV3;
  TV4 *= TV4;
  // Left edge
  if (NeedL) {
    float w0 = 0.039682539682539683f / (TV0 + 1.e-20f);
    float w1 = 0.31746031746031746f / (TV1 + 1.e-20f);
    float w2 = 0.47619047619047619f / (TV2 + 1.e-20f);
    float w3 = 0.15873015873015873f / (TV3 + 1.e-20f);
    float w4 = 0.0079365079365079365f / (TV4 + 1.e-20f);
    float rL = 1.0f / fmaxf(1.e-20f, w0 + w1 + w2 + w3 + w4);
    w0 *= rL;
    w1 *= rL;
    w2 *= rL;
    w3 *= rL;
    w4 *= rL;
    float L0 = -0.050000000000000000f * v[0] + 0.28333333333333333f * v[1] -
               0.71666666666666667f * v[2] + 1.2833333333333333f * v[3] +
               0.20000000000000000f * v[4];
    float L1 = 0.033333333333333333f * v[1] - 0.21666666666666667f * v[2] +
               0.78333333333333333f * v[3] + 0.45000000000000000f * v[4] -
               0.050000000000000000f * v[5];
    float L2 = -0.050000000000000000f * v[2] + 0.45000000000000000f * v[3] +
               0.78333333333333333f * v[4] - 0.21666666666666667f * v[5] +
               0.033333333333333333f * v[6];
    float L3 = 0.20000000000000000f * v[3] + 1.2833333333333333f * v[4] -
               0.71666666666666667f * v[5] + 0.28333333333333333f * v[6] -
               0.050000000000000000f * v[7];
    float L4 = 2.2833333333333333f * v[4] - 2.7166666666666667f * v[5] +
               2.2833333333333333f * v[6] - 1.0500000000000000f * v[7] +
               0.20000000000000000f * v[8];
    L = w0 * L0 + w1 * L1 + w2 * L2 + w3 * L3 + w4 * L4;
  }
  // Right edge
  if (NeedR) {
    float w0 = 0.0079365079365079365f / (TV0 + 1.e-20f);
    float w1 = 0.15873015873015873f / (TV1 + 1.e-20f);
    float w2 = 0.47619047619047619f / (TV2 + 1.e-20f);
    float w3 = 0.31746031746031746f / (TV3 + 1.e-20f);
    float w4 = 0.039682539682539683f / (TV4 + 1.e-20f);
    float rR = 1.0f / fmaxf(1.e-20f, w0 + w1 + w2 + w3 + w4);
    w0 *= rR;
    w1 *= rR;
    w2 *= rR;
    w3 *= rR;
    w4 *= rR;
    float R0 = 0.20000000000000000f * v[0] - 1.0500000000000000f * v[1] +
               2.2833333333333333f * v[2] - 2.7166666666666667f * v[3] +
               2.2833333333333333f * v[4];
    float R1 = -0.050000000000000000f * v[1] + 0.28333333333333333f * v[2] -
               0.71666666666666667f * v[3] + 1.2833333333333333f * v[4] +
               0.20000000000000000f * v[5];
    float R2 = 0.033333333333333333f * v[2] - 0.21666666666666667f * v[3] +
               0.78333333333333333f * v[4] + 0.45000000000000000f * v[5] -
               0.050000000000000000f * v[6];
    float R3 = -0.050000000000000000f * v[3] + 0.45000000000000000f * v[4] +
               0.78333333333333333f * v[5] - 0.21666666666666667f * v[6] +
               0.033333333333333333f * v[7];
    float R4 = 0.20000000000000000f * v[4] + 1.2833333333333333f * v[5] -
               0.71666666666666667f * v[6] + 0.28333333333333333f * v[7] -
               0.050000000000000000f * v[8];
    R = w0 * R0 + w1 * R1 + w2 * R2 + w3 * R3 + w4 * R4;
  }
}

// Full both-edge reconstruction (used by the advective sweeps, which consume vL
// and vR).
__host__ __device__ inline void weno9(const float v[ORD], float &L, float &R,
                                      bool immL, bool immR) {
  weno9_impl<true, true>(v, L, R, immL, immR);
}
// Single-edge helpers: compute only the left or only the right edge.
__host__ __device__ inline float weno9_left(const float v[ORD], bool immL,
                                            bool immR) {
  float L, R = 0.f;
  weno9_impl<true, false>(v, L, R, immL, immR);
  return L;
}
__host__ __device__ inline float weno9_right(const float v[ORD], bool immL,
                                             bool immR) {
  float L = 0.f, R;
  weno9_impl<false, true>(v, L, R, immL, immR);
  return R;
}

// Zero-derivative stencil modification at immersed boundaries (portUrb
// modify_stencil_immersed_der0). Once an immersed cell is encountered moving
// outward from the center, replicate the last in-domain value.
__host__ __device__ inline void
modify_stencil_immersed_der0(float s[ORD], const bool imm[ORD]) {
  constexpr int c = (ORD - 1) / 2; // stencil center (=4)
  if (!imm[c]) {
    for (int i2 = c + 1; i2 < ORD; i2++) {
      if (imm[i2]) {
        for (int i3 = i2; i3 < ORD; i3++)
          s[i3] = s[i2 - 1];
        break;
      }
    }
    for (int i2 = c - 1; i2 >= 0; i2--) {
      if (imm[i2]) {
        for (int i3 = i2; i3 >= 0; i3--)
          s[i3] = s[i2 + 1];
        break;
      }
    }
  }
}

// physics ACOUSTIC UPWINDING: edge pressure (p) and normal momentum (rmom) in
// each direction.

__host__ __device__ inline void
acoustic_x(int k, int j, int i, const Grid &g, const float *__restrict__ fields,
           const float *__restrict__ hy_dens,
           const float *__restrict__ immersed_prop, float &p_x, float &ru_x) {
  float s[ORD], pL, pR, ruL, ruR;
  const int kk = HS + k, jj = HS + j;
  bool imm[ORD];
  for (int ii = 0; ii < ORD; ii++)
    imm[ii] = immersed_prop[pidx(kk, jj, i + ii, g)] > IMM_TH;
  bool immL = imm[HSM1 - 1], immR = imm[HSM1 + 1];
  for (int ii = 0; ii < ORD; ii++)
    s[ii] = fields[fidx(idP, kk, jj, i + ii, g)];
  modify_stencil_immersed_der0(s, imm);
  pL = weno9_right(s, false, false);
  for (int ii = 0; ii < ORD; ii++)
    s[ii] = (fields[fidx(idR, kk, jj, i + ii, g)] + hy_dens[kk]) *
            fields[fidx(idU, kk, jj, i + ii, g)];
  ruL = weno9_right(s, immL, immR);
  for (int ii = 0; ii < ORD; ii++)
    imm[ii] = immersed_prop[pidx(kk, jj, i + ii + 1, g)] > IMM_TH;
  immL = imm[HSM1 - 1];
  immR = imm[HSM1 + 1];
  for (int ii = 0; ii < ORD; ii++)
    s[ii] = fields[fidx(idP, kk, jj, i + ii + 1, g)];
  modify_stencil_immersed_der0(s, imm);
  pR = weno9_left(s, false, false);
  for (int ii = 0; ii < ORD; ii++)
    s[ii] = (fields[fidx(idR, kk, jj, i + ii + 1, g)] + hy_dens[kk]) *
            fields[fidx(idU, kk, jj, i + ii + 1, g)];
  ruR = weno9_left(s, immL, immR);
  p_x = 0.5f * (pL + pR - g.cs * (ruR - ruL));
  ru_x = 0.5f * (ruL + ruR - (pR - pL) / g.cs);
}

__host__ __device__ inline void
acoustic_y(int k, int j, int i, const Grid &g, const float *__restrict__ fields,
           const float *__restrict__ hy_dens,
           const float *__restrict__ immersed_prop, float &p_y, float &rv_y) {
  float s[ORD], pL, pR, rvL, rvR;
  const int kk = HS + k, ii = HS + i;
  bool imm[ORD];
  for (int jj = 0; jj < ORD; jj++)
    imm[jj] = immersed_prop[pidx(kk, j + jj, ii, g)] > IMM_TH;
  bool immL = imm[HSM1 - 1], immR = imm[HSM1 + 1];
  for (int jj = 0; jj < ORD; jj++)
    s[jj] = fields[fidx(idP, kk, j + jj, ii, g)];
  modify_stencil_immersed_der0(s, imm);
  pL = weno9_right(s, false, false);
  for (int jj = 0; jj < ORD; jj++)
    s[jj] = (fields[fidx(idR, kk, j + jj, ii, g)] + hy_dens[kk]) *
            fields[fidx(idV, kk, j + jj, ii, g)];
  rvL = weno9_right(s, immL, immR);
  for (int jj = 0; jj < ORD; jj++)
    imm[jj] = immersed_prop[pidx(kk, j + jj + 1, ii, g)] > IMM_TH;
  immL = imm[HSM1 - 1];
  immR = imm[HSM1 + 1];
  for (int jj = 0; jj < ORD; jj++)
    s[jj] = fields[fidx(idP, kk, j + jj + 1, ii, g)];
  modify_stencil_immersed_der0(s, imm);
  pR = weno9_left(s, false, false);
  for (int jj = 0; jj < ORD; jj++)
    s[jj] = (fields[fidx(idR, kk, j + jj + 1, ii, g)] + hy_dens[kk]) *
            fields[fidx(idV, kk, j + jj + 1, ii, g)];
  rvR = weno9_left(s, immL, immR);
  p_y = 0.5f * (pL + pR - g.cs * (rvR - rvL));
  rv_y = 0.5f * (rvL + rvR - (pR - pL) / g.cs);
}

__host__ __device__ inline void
acoustic_z(int k, int j, int i, const Grid &g, const float *__restrict__ fields,
           const float *__restrict__ hy_dens,
           const float *__restrict__ immersed_prop, float &p_z, float &rw_z) {
  float s[ORD], pL, pR, rwL, rwR;
  const int jj = HS + j, ii = HS + i;
  bool imm[ORD];
  for (int kk = 0; kk < ORD; kk++)
    imm[kk] = immersed_prop[pidx(k + kk, jj, ii, g)] > IMM_TH;
  bool immL = imm[HSM1 - 1], immR = imm[HSM1 + 1];
  for (int kk = 0; kk < ORD; kk++)
    s[kk] = fields[fidx(idP, k + kk, jj, ii, g)];
  modify_stencil_immersed_der0(s, imm);
  pL = weno9_right(s, false, false);
  for (int kk = 0; kk < ORD; kk++)
    s[kk] = (fields[fidx(idR, k + kk, jj, ii, g)] + hy_dens[k + kk]) *
            fields[fidx(idW, k + kk, jj, ii, g)];
  rwL = weno9_right(s, immL, immR);
  for (int kk = 0; kk < ORD; kk++)
    imm[kk] = immersed_prop[pidx(k + kk + 1, jj, ii, g)] > IMM_TH;
  immL = imm[HSM1 - 1];
  immR = imm[HSM1 + 1];
  for (int kk = 0; kk < ORD; kk++)
    s[kk] = fields[fidx(idP, k + kk + 1, jj, ii, g)];
  modify_stencil_immersed_der0(s, imm);
  pR = weno9_left(s, false, false);
  for (int kk = 0; kk < ORD; kk++)
    s[kk] = (fields[fidx(idR, k + kk + 1, jj, ii, g)] + hy_dens[k + kk + 1]) *
            fields[fidx(idW, k + kk + 1, jj, ii, g)];
  rwR = weno9_left(s, immL, immR);
  p_z = 0.5f * (pL + pR - g.cs * (rwR - rwL));
  rw_z = 0.5f * (rwL + rwR - (pR - pL) / g.cs);
}

// ADVECTIVE UPWINDING: total edge fluxes for the NUM_STATE fields in each
// direction. For transverse velocities the immersed stencil is der0-modified
// and immL/immR are cleared.

__host__ __device__ inline void
advect_x(int k, int j, int i, const Grid &g, const float *__restrict__ fields,
         const float *__restrict__ hy_theta_cells,
         const float *__restrict__ immersed_prop, float ru, float p_x,
         float *__restrict__ flux_x) {
  const int Y = g.ny, X = g.nx + 1;
  const int kk = HS + k, jj = HS + j;
  const bool pos = ru > 0;
  const int ind = pos ? 0 : 1;
  float s[ORD];
  bool imm[ORD];
  for (int ii = 0; ii < ORD; ii++)
    imm[ii] = immersed_prop[pidx(kk, jj, i + ii + ind, g)] > IMM_TH;
  for (int l = idU; l < NUM_STATE; l++) {
    for (int ii = 0; ii < ORD; ii++)
      s[ii] = fields[fidx(l, kk, jj, i + ii + ind, g)];
    bool immL = imm[HSM1 - 1], immR = imm[HSM1 + 1];
    if (l == idV || l == idW) {
      modify_stencil_immersed_der0(s, imm);
      immL = false;
      immR = false;
    }
    float val = pos ? weno9_right(s, immL, immR) : weno9_left(s, immL, immR);
    if (l == idT)
      val += hy_theta_cells[kk];
    flux_x[x4(l, k, j, i, g.nz, Y, X)] = ru * val;
  }
  flux_x[x4(idR, k, j, i, g.nz, Y, X)] = ru;
  flux_x[x4(idU, k, j, i, g.nz, Y, X)] += p_x;
}

__host__ __device__ inline void
advect_y(int k, int j, int i, const Grid &g, const float *__restrict__ fields,
         const float *__restrict__ hy_theta_cells,
         const float *__restrict__ immersed_prop, float rv, float p_y,
         float *__restrict__ flux_y) {
  const int Y = g.ny + 1, X = g.nx;
  const int kk = HS + k, ii = HS + i;
  const bool pos = rv > 0;
  const int ind = pos ? 0 : 1;
  float s[ORD];
  bool imm[ORD];
  for (int jj = 0; jj < ORD; jj++)
    imm[jj] = immersed_prop[pidx(kk, j + jj + ind, ii, g)] > IMM_TH;
  for (int l = idU; l < NUM_STATE; l++) {
    for (int jj = 0; jj < ORD; jj++)
      s[jj] = fields[fidx(l, kk, j + jj + ind, ii, g)];
    bool immL = imm[HSM1 - 1], immR = imm[HSM1 + 1];
    if (l == idU || l == idW) {
      modify_stencil_immersed_der0(s, imm);
      immL = false;
      immR = false;
    }
    float val = pos ? weno9_right(s, immL, immR) : weno9_left(s, immL, immR);
    if (l == idT)
      val += hy_theta_cells[kk];
    flux_y[x4(l, k, j, i, g.nz, Y, X)] = rv * val;
  }
  flux_y[x4(idR, k, j, i, g.nz, Y, X)] = rv;
  flux_y[x4(idV, k, j, i, g.nz, Y, X)] += p_y;
}

__host__ __device__ inline void
advect_z(int k, int j, int i, const Grid &g, const float *__restrict__ fields,
         const float *__restrict__ hy_theta_edges,
         const float *__restrict__ immersed_prop, float rw, float p_z,
         float *__restrict__ flux_z) {
  const int Y = g.ny, X = g.nx;
  const int jj = HS + j, ii = HS + i;
  const bool pos = rw > 0;
  const int ind = pos ? 0 : 1;
  float s[ORD];
  bool imm[ORD];
  for (int kk = 0; kk < ORD; kk++)
    imm[kk] = immersed_prop[pidx(k + kk + ind, jj, ii, g)] > IMM_TH;
  for (int l = idU; l < NUM_STATE; l++) {
    for (int kk = 0; kk < ORD; kk++)
      s[kk] = fields[fidx(l, k + kk + ind, jj, ii, g)];
    bool immL = imm[HSM1 - 1], immR = imm[HSM1 + 1];
    if (l == idU || l == idV) {
      modify_stencil_immersed_der0(s, imm);
      immL = false;
      immR = false;
    }
    float val = pos ? weno9_right(s, immL, immR) : weno9_left(s, immL, immR);
    if (l == idT)
      val += hy_theta_edges[k];
    flux_z[x4(l, k, j, i, g.nz + 1, Y, X)] = rw * val;
  }
  flux_z[x4(idR, k, j, i, g.nz + 1, Y, X)] = rw;
  flux_z[x4(idW, k, j, i, g.nz + 1, Y, X)] += p_z;
}

// FLUX DIVERGENCE + gravity source -> state tendency for field l.
__host__ __device__ inline float
tendency(int l, int k, int j, int i, const Grid &g,
         const float *__restrict__ fields, const float *__restrict__ flux_x,
         const float *__restrict__ flux_y, const float *__restrict__ flux_z) {
  const int Xx = g.nx + 1, Yy = g.ny;  // flux_x dims (nz, ny, nx+1)
  const int Xy = g.nx, Yy2 = g.ny + 1; // flux_y dims (nz, ny+1, nx)
  float t = -(flux_x[x4(l, k, j, i + 1, g.nz, Yy, Xx)] -
              flux_x[x4(l, k, j, i, g.nz, Yy, Xx)]) *
                g.r_dx -
            (flux_y[x4(l, k, j + 1, i, g.nz, Yy2, Xy)] -
             flux_y[x4(l, k, j, i, g.nz, Yy2, Xy)]) *
                g.r_dy -
            (flux_z[x4(l, k + 1, j, i, g.nz + 1, g.ny, g.nx)] -
             flux_z[x4(l, k, j, i, g.nz + 1, g.ny, g.nx)]) *
                g.r_dz;
  if (l == idW)
    t += -g.grav * fields[fidx(idR, HS + k, HS + j, HS + i, g)];
  return t;
}

#endif // WENOFV_KERNELS_H
