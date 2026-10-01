#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <omp.h>

#pragma omp declare target
#include "kernels.h"
#pragma omp end declare target
#include "reference.h"

#define TPB 256

int main(int argc, char **argv) {
  if (argc != 5) {
    printf("Usage: %s <nx> <ny> <nz> <repeat>\n", argv[0]);
    printf("  e.g. %s 256 256 64 100\n", argv[0]);
    return 1;
  }
  Grid g;
  g.nx = atoi(argv[1]);
  g.ny = atoi(argv[2]);
  g.nz = atoi(argv[3]);
  const int repeat = atoi(argv[4]);
  g.px = g.nx + 2 * HS;
  g.py = g.ny + 2 * HS;
  g.pz = g.nz + 2 * HS;
  const float dx = 20.f;
  g.cs = 350.f;
  g.grav = 9.81f;
  g.r_dx = 1.f / dx;
  g.r_dy = 1.f / dx;
  g.r_dz = 1.f / dx;

  printf("Grid: %d x %d x %d (interior), WENO order %d, halo %d\n", g.nx, g.ny,
         g.nz, ORD, HS);

  const size_t nfields = (size_t)NV * g.pz * g.py * g.px;
  const size_t ntend = (size_t)NUM_STATE * g.nz * g.ny * g.nx;
  const size_t nprop = (size_t)g.pz * g.py * g.px;
  std::vector<float> h_fields(nfields);
  std::vector<float> h_hy_dens(g.pz), h_hy_theta_cells(g.pz),
      h_hy_theta_edges(g.nz + 1);
  std::vector<float> h_imm(nprop, 0.f);
  std::vector<float> h_tend(ntend), h_tend_ref(ntend);

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dR(-0.1f, 0.1f), dUVW(-8.f, 8.f),
      dTh(-2.f, 2.f), dP(-200.f, 200.f);
  for (int v = 0; v < NV; v++) {
    for (int k = 0; k < g.pz; k++)
      for (int j = 0; j < g.py; j++)
        for (int i = 0; i < g.px; i++) {
          float val;
          switch (v) {
          case idR:
            val = dR(rng);
            break;
          case idU:
          case idV:
          case idW:
            val = dUVW(rng);
            break;
          case idT:
            val = dTh(rng);
            break;
          default:
            val = dP(rng);
            break; // idP
          }
          h_fields[fidx(v, k, j, i, g)] = val;
        }
  }
  for (int k = 0; k < g.pz; k++) {
    float z = (k - HS) * dx;
    h_hy_dens[k] = 1.2f * expf(-fmaxf(0.f, z) / 8000.f);
    h_hy_theta_cells[k] = 300.f + 0.01f * z;
  }
  for (int k = 0; k <= g.nz; k++)
    h_hy_theta_edges[k] = 300.f + 0.01f * (k * dx);

  // Synthesize a rectangular building obstacle centered in x/y, sitting on the
  // ground. Immersed proportion is 1 inside the box, 0 outside (interior
  // region, offset by halos).
  const int bx0 = g.nx / 2 - g.nx / 8, bx1 = g.nx / 2 + g.nx / 8;
  const int by0 = g.ny / 2 - g.ny / 8, by1 = g.ny / 2 + g.ny / 8;
  const int bz1 = g.nz / 4;
  size_t nimm = 0;
  for (int k = 0; k < g.nz; k++)
    for (int j = 0; j < g.ny; j++)
      for (int i = 0; i < g.nx; i++) {
        if (i >= bx0 && i < bx1 && j >= by0 && j < by1 && k < bz1) {
          h_imm[pidx(HS + k, HS + j, HS + i, g)] = 1.f;
          nimm++;
        }
      }
  printf("Immersed building: %zu / %d cells (%.1f%%)\n", nimm,
         g.nx * g.ny * g.nz, 100.0 * nimm / ((double)g.nx * g.ny * g.nz));

  reference(g, h_fields.data(), h_hy_dens.data(), h_hy_theta_cells.data(),
            h_hy_theta_edges.data(), h_imm.data(), h_tend_ref.data());

  const size_t nfx = (size_t)NUM_STATE * g.nz * g.ny * (g.nx + 1);
  const size_t nfy = (size_t)NUM_STATE * g.nz * (g.ny + 1) * g.nx;
  const size_t nfz = (size_t)NUM_STATE * (g.nz + 1) * g.ny * g.nx;
  const size_t nex = (size_t)g.nz * g.ny * (g.nx + 1);
  const size_t ney = (size_t)g.nz * (g.ny + 1) * g.nx;
  const size_t nez = (size_t)(g.nz + 1) * g.ny * g.nx;

  // Device-resident scratch buffers (edge quantities and directional fluxes).
  std::vector<float> h_px(nex), h_rux(nex), h_py(ney), h_rvy(ney), h_pz(nez),
      h_rwz(nez);
  std::vector<float> h_fx(nfx), h_fy(nfy), h_fz(nfz);

  const float *fields = h_fields.data();
  const float *hy_dens = h_hy_dens.data();
  const float *hy_tc = h_hy_theta_cells.data();
  const float *hy_te = h_hy_theta_edges.data();
  const float *imm = h_imm.data();
  float *px = h_px.data();
  float *rux = h_rux.data();
  float *py = h_py.data();
  float *rvy = h_rvy.data();
  float *pz = h_pz.data();
  float *rwz = h_rwz.data();
  float *fx = h_fx.data();
  float *fy = h_fy.data();
  float *fz = h_fz.data();
  float *tend = h_tend.data();

#pragma omp target data map(to : fields[0 : nfields], hy_dens[0 : g.pz],       \
                                hy_tc[0 : g.pz], hy_te[0 : g.nz + 1],          \
                                imm[0 : nprop])                                \
                        map(from : tend[0 : ntend])                            \
    map(alloc : px[0 : nex], rux[0 : nex], py[0 : ney], rvy[0 : ney],          \
                pz[0 : nez], rwz[0 : nez], fx[0 : nfx], fy[0 : nfy], fz[0 : nfz])
  {
    // One invocation of the 7-kernel compute_tendencies sequence.
    auto run_kernels = [&]() {
      // acoustic x
      #pragma omp target teams distribute parallel for
      for (size_t t = 0; t < nex; t++) {
        int i = t % (g.nx + 1);
        size_t r = t / (g.nx + 1);
        int j = r % g.ny;
        int k = r / g.ny;
        acoustic_x(k, j, i, g, fields, hy_dens, imm,
                   px[x3(k, j, i, g.ny, g.nx + 1)],
                   rux[x3(k, j, i, g.ny, g.nx + 1)]);
      }
      // acoustic y
      #pragma omp target teams distribute parallel for
      for (size_t t = 0; t < ney; t++) {
        int i = t % g.nx;
        size_t r = t / g.nx;
        int j = r % (g.ny + 1);
        int k = r / (g.ny + 1);
        acoustic_y(k, j, i, g, fields, hy_dens, imm,
                   py[x3(k, j, i, g.ny + 1, g.nx)],
                   rvy[x3(k, j, i, g.ny + 1, g.nx)]);
      }
      // acoustic z
      #pragma omp target teams distribute parallel for
      for (size_t t = 0; t < nez; t++) {
        int i = t % g.nx;
        size_t r = t / g.nx;
        int j = r % g.ny;
        int k = r / g.ny;
        acoustic_z(k, j, i, g, fields, hy_dens, imm,
                   pz[x3(k, j, i, g.ny, g.nx)], rwz[x3(k, j, i, g.ny, g.nx)]);
      }
      // advect x
      #pragma omp target teams distribute parallel for
      for (size_t t = 0; t < nex; t++) {
        int i = t % (g.nx + 1);
        size_t r = t / (g.nx + 1);
        int j = r % g.ny;
        int k = r / g.ny;
        size_t e = x3(k, j, i, g.ny, g.nx + 1);
        advect_x(k, j, i, g, fields, hy_tc, imm, rux[e], px[e], fx);
      }
      // advect y
      #pragma omp target teams distribute parallel for
      for (size_t t = 0; t < ney; t++) {
        int i = t % g.nx;
        size_t r = t / g.nx;
        int j = r % (g.ny + 1);
        int k = r / (g.ny + 1);
        size_t e = x3(k, j, i, g.ny + 1, g.nx);
        advect_y(k, j, i, g, fields, hy_tc, imm, rvy[e], py[e], fy);
      }
      // advect z
      #pragma omp target teams distribute parallel for
      for (size_t t = 0; t < nez; t++) {
        int i = t % g.nx;
        size_t r = t / g.nx;
        int j = r % g.ny;
        int k = r / g.ny;
        size_t e = x3(k, j, i, g.ny, g.nx);
        advect_z(k, j, i, g, fields, hy_te, imm, rwz[e], pz[e], fz);
      }
      // tendency
      #pragma omp target teams distribute parallel for
      for (size_t t = 0; t < ntend; t++) {
        int i = t % g.nx;
        size_t r = t / g.nx;
        int j = r % g.ny;
        r /= g.ny;
        int k = r % g.nz;
        int l = r / g.nz;
        tend[x4(l, k, j, i, g.nz, g.ny, g.nx)] =
            tendency(l, k, j, i, g, fields, fx, fy, fz);
      }
    };

    // Warmup
    const int warmup = 100;
    for (int r = 0; r < warmup; r++)
      run_kernels();

    auto start = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; r++)
      run_kernels();
    auto end = std::chrono::steady_clock::now();
    double us =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
            .count() * 1e-3;
    printf(
        "Average execution time of compute_tendencies (7 kernels): %f (us)\n",
        us / repeat);
  }

  double max_rel = 0.0;
  size_t nbad = 0;
  for (size_t n = 0; n < ntend; n++) {
    float a = h_tend[n], b = h_tend_ref[n];
    double denom = fmax(1e-4, fmax(fabs((double)a), fabs((double)b)));
    double rel = fabs((double)a - (double)b) / denom;
    if (rel > max_rel)
      max_rel = rel;
    if (rel > 1e-2)
      nbad++;
  }
  printf("Max relative error: %e (%zu / %zu elements exceed 1e-2)\n", max_rel,
         nbad, ntend);
  printf("%s\n", nbad == 0 ? "PASS" : "FAIL");

  return 0;
}
