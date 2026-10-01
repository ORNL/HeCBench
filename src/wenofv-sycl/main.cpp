#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <sycl/sycl.hpp>
#include "kernels.h"
#include "reference.h"

#define TPB 256

// Round a global size up to a whole multiple of the local size.
static inline size_t round_up(size_t n, size_t l) {
  return (n + l - 1) / l * l;
}

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

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_fields = sycl::malloc_device<float>(nfields, q);
  float *d_hy_dens = sycl::malloc_device<float>(g.pz, q);
  float *d_hy_tc = sycl::malloc_device<float>(g.pz, q);
  float *d_hy_te = sycl::malloc_device<float>(g.nz + 1, q);
  float *d_imm = sycl::malloc_device<float>(nprop, q);
  float *d_px = sycl::malloc_device<float>(nex, q);
  float *d_rux = sycl::malloc_device<float>(nex, q);
  float *d_py = sycl::malloc_device<float>(ney, q);
  float *d_rvy = sycl::malloc_device<float>(ney, q);
  float *d_pz = sycl::malloc_device<float>(nez, q);
  float *d_rwz = sycl::malloc_device<float>(nez, q);
  float *d_fx = sycl::malloc_device<float>(nfx, q);
  float *d_fy = sycl::malloc_device<float>(nfy, q);
  float *d_fz = sycl::malloc_device<float>(nfz, q);
  float *d_tend = sycl::malloc_device<float>(ntend, q);

  q.memcpy(d_fields, h_fields.data(), nfields * sizeof(float));
  q.memcpy(d_hy_dens, h_hy_dens.data(), g.pz * sizeof(float));
  q.memcpy(d_hy_tc, h_hy_theta_cells.data(), g.pz * sizeof(float));
  q.memcpy(d_hy_te, h_hy_theta_edges.data(), (g.nz + 1) * sizeof(float));
  q.memcpy(d_imm, h_imm.data(), nprop * sizeof(float));

  const sycl::range<1> lws(TPB);

  auto run_kernels = [&]() {
    // acoustic x
    q.submit([&](sycl::handler &h) {
      sycl::range<1> gws(round_up(nex, TPB));
      h.parallel_for<class acoustic_x_k>(
          sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> it) {
            size_t t = it.get_global_id(0);
            if (t >= nex)
              return;
            int i = t % (g.nx + 1);
            size_t rr = t / (g.nx + 1);
            int j = rr % g.ny;
            int k = rr / g.ny;
            acoustic_x(k, j, i, g, d_fields, d_hy_dens, d_imm,
                       d_px[x3(k, j, i, g.ny, g.nx + 1)],
                       d_rux[x3(k, j, i, g.ny, g.nx + 1)]);
          });
    });
    // acoustic y
    q.submit([&](sycl::handler &h) {
      sycl::range<1> gws(round_up(ney, TPB));
      h.parallel_for<class acoustic_y_k>(
          sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> it) {
            size_t t = it.get_global_id(0);
            if (t >= ney)
              return;
            int i = t % g.nx;
            size_t rr = t / g.nx;
            int j = rr % (g.ny + 1);
            int k = rr / (g.ny + 1);
            acoustic_y(k, j, i, g, d_fields, d_hy_dens, d_imm,
                       d_py[x3(k, j, i, g.ny + 1, g.nx)],
                       d_rvy[x3(k, j, i, g.ny + 1, g.nx)]);
          });
    });
    // acoustic z
    q.submit([&](sycl::handler &h) {
      sycl::range<1> gws(round_up(nez, TPB));
      h.parallel_for<class acoustic_z_k>(
          sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> it) {
            size_t t = it.get_global_id(0);
            if (t >= nez)
              return;
            int i = t % g.nx;
            size_t rr = t / g.nx;
            int j = rr % g.ny;
            int k = rr / g.ny;
            acoustic_z(k, j, i, g, d_fields, d_hy_dens, d_imm,
                       d_pz[x3(k, j, i, g.ny, g.nx)],
                       d_rwz[x3(k, j, i, g.ny, g.nx)]);
          });
    });
    // advect x
    q.submit([&](sycl::handler &h) {
      sycl::range<1> gws(round_up(nex, TPB));
      h.parallel_for<class advect_x_k>(
          sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> it) {
            size_t t = it.get_global_id(0);
            if (t >= nex)
              return;
            int i = t % (g.nx + 1);
            size_t rr = t / (g.nx + 1);
            int j = rr % g.ny;
            int k = rr / g.ny;
            size_t e = x3(k, j, i, g.ny, g.nx + 1);
            advect_x(k, j, i, g, d_fields, d_hy_tc, d_imm, d_rux[e], d_px[e],
                     d_fx);
          });
    });
    // advect y
    q.submit([&](sycl::handler &h) {
      sycl::range<1> gws(round_up(ney, TPB));
      h.parallel_for<class advect_y_k>(
          sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> it) {
            size_t t = it.get_global_id(0);
            if (t >= ney)
              return;
            int i = t % g.nx;
            size_t rr = t / g.nx;
            int j = rr % (g.ny + 1);
            int k = rr / (g.ny + 1);
            size_t e = x3(k, j, i, g.ny + 1, g.nx);
            advect_y(k, j, i, g, d_fields, d_hy_tc, d_imm, d_rvy[e], d_py[e],
                     d_fy);
          });
    });
    // advect z
    q.submit([&](sycl::handler &h) {
      sycl::range<1> gws(round_up(nez, TPB));
      h.parallel_for<class advect_z_k>(
          sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> it) {
            size_t t = it.get_global_id(0);
            if (t >= nez)
              return;
            int i = t % g.nx;
            size_t rr = t / g.nx;
            int j = rr % g.ny;
            int k = rr / g.ny;
            size_t e = x3(k, j, i, g.ny, g.nx);
            advect_z(k, j, i, g, d_fields, d_hy_te, d_imm, d_rwz[e], d_pz[e],
                     d_fz);
          });
    });
    // tendency
    q.submit([&](sycl::handler &h) {
      sycl::range<1> gws(round_up(ntend, TPB));
      h.parallel_for<class tendency_k>(
          sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> it) {
            size_t t = it.get_global_id(0);
            if (t >= ntend)
              return;
            int i = t % g.nx;
            size_t rr = t / g.nx;
            int j = rr % g.ny;
            rr /= g.ny;
            int k = rr % g.nz;
            int l = rr / g.nz;
            d_tend[x4(l, k, j, i, g.nz, g.ny, g.nx)] =
                tendency(l, k, j, i, g, d_fields, d_fx, d_fy, d_fz);
          });
    });
  };

  // Warmup
  const int warmup = 100;
  for (int r = 0; r < warmup; r++)
    run_kernels();
  q.wait();

  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++)
    run_kernels();
  q.wait();
  auto end = std::chrono::steady_clock::now();
  double us = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
                  .count() * 1e-3;
  printf("Average execution time of compute_tendencies (7 kernels): %f (us)\n",
         us / repeat);

  q.memcpy(h_tend.data(), d_tend, ntend * sizeof(float)).wait();

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

  sycl::free(d_fields, q);
  sycl::free(d_hy_dens, q);
  sycl::free(d_hy_tc, q);
  sycl::free(d_hy_te, q);
  sycl::free(d_imm, q);
  sycl::free(d_px, q);
  sycl::free(d_rux, q);
  sycl::free(d_py, q);
  sycl::free(d_rvy, q);
  sycl::free(d_pz, q);
  sycl::free(d_rwz, q);
  sycl::free(d_fx, q);
  sycl::free(d_fy, q);
  sycl::free(d_fz, q);
  sycl::free(d_tend, q);
  return 0;
}
