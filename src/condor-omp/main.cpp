// OpenMP port of the Condor dense-temperature benchmark.
//
// OpenMP has no portable wave size. The cooperative kernel vectorizes the
// snapshot loop with a simd reduction. The second kernel gives each thread a
// short run of points along z, which share x and y, and overlaps two snapshots
// so their exp evaluations stay independent. --block is the target team thread limit.
//
#include "condor_host.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <omp.h>

// Temperature at every grid point. The simd reduction splits the beam snapshots
// across vector lanes. OpenMP chooses the lane count.
void dense_temperature_team(const double* xp, const double* yp, const double* zp, const double* xb,
                            const double* yb, const double* zb, const double* phix, const double* phiy,
                            const double* phiz, const double* dtau, const double* expmod, int npoints,
                            int nnodes, double t_init, double* temperature, int block) {
#pragma omp target teams distribute parallel for thread_limit(block)
  for (int p = 0; p < npoints; ++p) {
    const double x = xp[p];
    const double y = yp[p];
    const double z = zp[p];
    double dT = 0.0;
#pragma omp simd reduction(+ : dT)
    for (int node = 0; node < nnodes; ++node) {
      dT += snapshot_heat(x, y, z, xb[node], yb[node], zb[node], phix[node], phiy[node], phiz[node],
                          dtau[node], expmod[node]);
    }
    temperature[p] = t_init + dT;
  }
}

// Points along z that one thread keeps together. They share x and y.
constexpr int kZPoints = 4;

// Heat of one snapshot at those z points. The x and y terms match snapshot_heat
// and are computed once. Each phi is written before it is added, so two calls
// can evaluate exp independently.
__attribute__((always_inline)) inline void snapshot_phi_z(double x, double y, const double* z, double* phi,
                                                         double xb, double yb, double zb, double phix,
                                                         double phiy, double phiz, double expmod) {
  const double dx = x - xb;
  const double dy = y - yb;
  const double xy = (dx * dx * phix) + (dy * dy * phiy);
  for (int t = 0; t < kZPoints; ++t) {
    const double dz = z[t] - zb;
    const double exponent = -3.0 * (xy + (dz * dz * phiz)) + expmod;
    phi[t] = (exponent < kExpRoundsToZero) ? 0.0 : exp(exponent);
  }
}

// Same temperature field. Each thread owns four consecutive z points and applies
// two beam snapshots at a time. A depth that is not a multiple of four uses the
// one-point loop for the leftover points.
void dense_temperature_ztile(const double* xp, const double* yp, const double* zp, const double* xb,
                             const double* yb, const double* zb, const double* phix, const double* phiy,
                             const double* phiz, const double* dtau, const double* expmod, int npoints,
                             int nnodes, double t_init, double* temperature, int nz, int block) {
  const int nrows = npoints / nz;
  const int ntiles = nz / kZPoints;
  const int nrem = nz % kZPoints;

  if (ntiles > 0) {
#pragma omp target teams distribute parallel for collapse(2) thread_limit(block)
    for (int row = 0; row < nrows; ++row) {
      for (int tile = 0; tile < ntiles; ++tile) {
        const int p0 = row * nz + tile * kZPoints;
        const double x = xp[p0];
        const double y = yp[p0];
        double z[kZPoints];
        double dT[kZPoints];
        for (int t = 0; t < kZPoints; ++t) {
          z[t] = zp[p0 + t];
          dT[t] = 0.0;
        }
        int node = 0;
        for (; node + 1 < nnodes; node += 2) {
          double phi0[kZPoints];
          double phi1[kZPoints];
          snapshot_phi_z(x, y, z, phi0, xb[node], yb[node], zb[node], phix[node], phiy[node], phiz[node],
                         expmod[node]);
          snapshot_phi_z(x, y, z, phi1, xb[node + 1], yb[node + 1], zb[node + 1], phix[node + 1],
                         phiy[node + 1], phiz[node + 1], expmod[node + 1]);
          const double dtau0 = dtau[node];
          const double dtau1 = dtau[node + 1];
          for (int t = 0; t < kZPoints; ++t) {
            dT[t] += dtau0 * phi0[t];
            dT[t] += dtau1 * phi1[t];
          }
        }
        if (node < nnodes) {
          double phi[kZPoints];
          snapshot_phi_z(x, y, z, phi, xb[node], yb[node], zb[node], phix[node], phiy[node], phiz[node],
                         expmod[node]);
          const double w = dtau[node];
          for (int t = 0; t < kZPoints; ++t) {
            dT[t] += w * phi[t];
          }
        }
        for (int t = 0; t < kZPoints; ++t) {
          temperature[p0 + t] = t_init + dT[t];
        }
      }
    }
  }

  if (nrem > 0) {
#pragma omp target teams distribute parallel for collapse(2) thread_limit(block)
    for (int row = 0; row < nrows; ++row) {
      for (int k = nz - nrem; k < nz; ++k) {
        const int p = row * nz + k;
        const double x = xp[p];
        const double y = yp[p];
        const double z = zp[p];
        double dT = 0.0;
        int node = 0;
        for (; node + 1 < nnodes; node += 2) {
          dT += snapshot_heat(x, y, z, xb[node], yb[node], zb[node], phix[node], phiy[node], phiz[node],
                              dtau[node], expmod[node]);
          dT += snapshot_heat(x, y, z, xb[node + 1], yb[node + 1], zb[node + 1], phix[node + 1],
                              phiy[node + 1], phiz[node + 1], dtau[node + 1], expmod[node + 1]);
        }
        if (node < nnodes) {
          dT += snapshot_heat(x, y, z, xb[node], yb[node], zb[node], phix[node], phiy[node], phiz[node],
                              dtau[node], expmod[node]);
        }
        temperature[p] = t_init + dT;
      }
    }
  }
}

// Check both device fields against the host, then report the average time of each
// kernel.
int run(const BenchConfig& cfg) {
  const int nx = cfg.nx;
  const int ny = cfg.ny;
  const int nz = cfg.nz;
  const int nnodes = cfg.nnodes;
  const int iters = cfg.iters;
  const int warmup = cfg.warmup;
  const int block = cfg.block;

  Nodes nodes;
  Grid grid;
  build_scan(nodes, nnodes);
  build_grid(grid, nx, ny, nz);
  constexpr double kTInit = 300.0;
  const int npoints = nx * ny * nz;

  const double* xp = grid.xp.data();
  const double* yp = grid.yp.data();
  const double* zp = grid.zp.data();
  const double* xb = nodes.xb.data();
  const double* yb = nodes.yb.data();
  const double* zb = nodes.zb.data();
  const double* phix = nodes.phix.data();
  const double* phiy = nodes.phiy.data();
  const double* phiz = nodes.phiz.data();
  const double* dtau = nodes.dtau.data();
  const double* expmod = nodes.expmod.data();
  std::vector<double> host_T(static_cast<size_t>(npoints));
  double* temperature = host_T.data();

  double simd_abs = 0.0;
  double simd_rel = 0.0;
  double ztile_abs = 0.0;
  double ztile_rel = 0.0;
  bool simd_pass = false;
  bool ztile_pass = false;
  double simd_ms = 0.0;
  double ztile_ms = 0.0;

#pragma omp target data map(to : xp[0 : npoints], yp[0 : npoints], zp[0 : npoints], xb[0 : nnodes], \
                                yb[0 : nnodes], zb[0 : nnodes], phix[0 : nnodes], phiy[0 : nnodes],  \
                                phiz[0 : nnodes], dtau[0 : nnodes], expmod[0 : nnodes])             \
    map(alloc : temperature[0 : npoints])
  {
    dense_temperature_team(xp, yp, zp, xb, yb, zb, phix, phiy, phiz, dtau, expmod, npoints, nnodes, kTInit,
                           temperature, block);
#pragma omp target update from(temperature[0 : npoints])
    simd_pass = compare_temperature(host_T, nodes, grid, kTInit, simd_abs, simd_rel);

    dense_temperature_ztile(xp, yp, zp, xb, yb, zb, phix, phiy, phiz, dtau, expmod, npoints, nnodes, kTInit,
                            temperature, nz, block);
#pragma omp target update from(temperature[0 : npoints])
    ztile_pass = compare_temperature(host_T, nodes, grid, kTInit, ztile_abs, ztile_rel);

    printf("device: OpenMP  devices: %d  threads: %d\n", omp_get_num_devices(), omp_get_max_threads());
    printf("grid: %d x %d x %d (%d points)  nodes: %d  block: %d\n", nx, ny, nz, npoints, nnodes, block);
    printf("warmup: %d  iters: %d\n", warmup, iters);
    printf("simd max abs error: %.3e  max rel error: %.3e\n", simd_abs, simd_rel);
    printf("z-tile max abs error: %.3e  max rel error: %.3e\n", ztile_abs, ztile_rel);

    if (simd_pass && ztile_pass) {
      for (int i = 0; i < warmup; ++i) {
        dense_temperature_team(xp, yp, zp, xb, yb, zb, phix, phiy, phiz, dtau, expmod, npoints, nnodes,
                               kTInit, temperature, block);
      }
      const auto simd_start = std::chrono::steady_clock::now();
      for (int i = 0; i < iters; ++i) {
        dense_temperature_team(xp, yp, zp, xb, yb, zb, phix, phiy, phiz, dtau, expmod, npoints, nnodes,
                               kTInit, temperature, block);
      }
      const auto simd_stop = std::chrono::steady_clock::now();
      simd_ms = std::chrono::duration<double, std::milli>(simd_stop - simd_start).count() /
                static_cast<double>(iters);

      for (int i = 0; i < warmup; ++i) {
        dense_temperature_ztile(xp, yp, zp, xb, yb, zb, phix, phiy, phiz, dtau, expmod, npoints, nnodes,
                                kTInit, temperature, nz, block);
      }
      const auto ztile_start = std::chrono::steady_clock::now();
      for (int i = 0; i < iters; ++i) {
        dense_temperature_ztile(xp, yp, zp, xb, yb, zb, phix, phiy, phiz, dtau, expmod, npoints, nnodes,
                                kTInit, temperature, nz, block);
      }
      const auto ztile_stop = std::chrono::steady_clock::now();
      ztile_ms = std::chrono::duration<double, std::milli>(ztile_stop - ztile_start).count() /
                 static_cast<double>(iters);
    }
  }

  if (!simd_pass || !ztile_pass) {
    printf("FAIL\n");
    return EXIT_FAILURE;
  }
  printf("Average kernel time (simd): %.4f ms\n", simd_ms);
  printf("Average kernel time (z-tile): %.4f ms\n", ztile_ms);
  printf("PASS\n");
  return EXIT_SUCCESS;
}

// Parse the grid size, beam-snapshot count, and launch settings, then run.
int main(int argc, char** argv) {
  return run(parse_options(argc, argv));
}
