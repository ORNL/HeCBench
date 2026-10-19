// Dense temperature field from ORNL Condor.
//
// Condor evaluates an ellipsoidal Gaussian volumetric heat source by summing
// prepared quadrature nodes at every grid point (Stump and Plotkowski, 2019;
// Nguyen et al., Welding Journal, 1999, Eq. 7). The device kernel is
// Calculate_Dense_Temperatures / T_device: one team per point, a reduction
// over nodes, and a single write of T_init + dT. A second, warp-agnostic
// kernel assigns one thread to each grid point. A third kernel gives each
// thread four consecutive points along z and is timed for comparison.
//
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#include "condor_host.hpp"

#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t err_ = (expr);                                                 \
    if (err_ != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error %s:%d %s\n", __FILE__, __LINE__,             \
              cudaGetErrorString(err_));                                       \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// Add one double across the 32 lanes of this warp. Only lane 0 receives the total.
__device__ __forceinline__ double warp_sum(double value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

// Temperature at every grid point. One warp owns one point and splits the beam
// snapshots across its lanes. Lane 0 writes T_init plus the summed contributions.
// Several warps share a block so a warp does not need a block of its own.
__global__ void dense_temperature_team(const double* __restrict__ xp, const double* __restrict__ yp,
                                       const double* __restrict__ zp, const double* __restrict__ xb,
                                       const double* __restrict__ yb, const double* __restrict__ zb,
                                       const double* __restrict__ phix, const double* __restrict__ phiy,
                                       const double* __restrict__ phiz, const double* __restrict__ dtau,
                                       const double* __restrict__ expmod, int npoints, int nnodes,
                                       double t_init, double* __restrict__ temperature) {
  constexpr int kWarpSize = 32;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x / kWarpSize;
  const int warps_per_block = blockDim.x / kWarpSize;
  const int p = blockIdx.x * warps_per_block + warp;
  if (p >= npoints) {
    return;
  }
  const double x = xp[p];
  const double y = yp[p];
  const double z = zp[p];

  double dT = 0.0;
  for (int node = lane; node < nnodes; node += kWarpSize) {
    dT += snapshot_heat(x, y, z, xb[node], yb[node], zb[node], phix[node], phiy[node], phiz[node],
                        dtau[node], expmod[node]);
  }
  dT = warp_sum(dT);
  if (lane == 0) {
    temperature[p] = t_init + dT;
  }
}

// Same temperature field, with one thread per grid point. The thread visits the
// beam snapshots in order and uses no warp size or warp shuffle, so the mapping
// is valid for any warp width.
__global__ void dense_temperature_point(const double* __restrict__ xp, const double* __restrict__ yp,
                                        const double* __restrict__ zp, const double* __restrict__ xb,
                                        const double* __restrict__ yb, const double* __restrict__ zb,
                                        const double* __restrict__ phix, const double* __restrict__ phiy,
                                        const double* __restrict__ phiz, const double* __restrict__ dtau,
                                        const double* __restrict__ expmod, int npoints, int nnodes,
                                        double t_init, double* __restrict__ temperature) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= npoints) {
    return;
  }
  const double x = xp[p];
  const double y = yp[p];
  const double z = zp[p];
  double dT = 0.0;
  for (int node = 0; node < nnodes; ++node) {
    dT += snapshot_heat(x, y, z, xb[node], yb[node], zb[node], phix[node], phiy[node], phiz[node],
                        dtau[node], expmod[node]);
  }
  temperature[p] = t_init + dT;
}

// Points along z that one thread keeps together. They share x and y.
constexpr int kZPoints = 4;

// Heat of one snapshot at those z points. The x and y terms match snapshot_heat
// and are computed once. Each phi is written before it is added, so two calls
// can evaluate exp independently.
__device__ __forceinline__ void snapshot_phi_z(double x, double y, const double* z, double* phi, double xb,
                                               double yb, double zb, double phix, double phiy, double phiz,
                                               double expmod) {
  const double dx = x - xb;
  const double dy = y - yb;
  const double xy = (dx * dx * phix) + (dy * dy * phiy);
#pragma unroll
  for (int t = 0; t < kZPoints; ++t) {
    const double dz = z[t] - zb;
    const double exponent = -3.0 * (xy + (dz * dz * phiz)) + expmod;
    phi[t] = (exponent < kExpRoundsToZero) ? 0.0 : exp(exponent);
  }
}

// Same temperature field. Each thread owns four consecutive z points and applies
// two beam snapshots at a time.
__global__ void dense_temperature_ztile(const double* __restrict__ xp, const double* __restrict__ yp,
                                        const double* __restrict__ zp, const double* __restrict__ xb,
                                        const double* __restrict__ yb, const double* __restrict__ zb,
                                        const double* __restrict__ phix, const double* __restrict__ phiy,
                                        const double* __restrict__ phiz, const double* __restrict__ dtau,
                                        const double* __restrict__ expmod, int npoints, int nnodes, int nz,
                                        double t_init, double* __restrict__ temperature) {
  const int ntiles = nz / kZPoints;
  const int nrows = npoints / nz;
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= nrows * ntiles) {
    return;
  }
  const int row = id / ntiles;
  const int tile = id - row * ntiles;
  const int p0 = row * nz + tile * kZPoints;
  const double x = xp[p0];
  const double y = yp[p0];
  double z[kZPoints];
  double dT[kZPoints];
#pragma unroll
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
#pragma unroll
    for (int t = 0; t < kZPoints; ++t) {
      dT[t] += dtau0 * phi0[t];
      dT[t] += dtau1 * phi1[t];
    }
  }
  if (node < nnodes) {
    double phi[kZPoints];
    snapshot_phi_z(x, y, z, phi, xb[node], yb[node], zb[node], phix[node], phiy[node], phiz[node],
                   expmod[node]);
    const double weight = dtau[node];
#pragma unroll
    for (int t = 0; t < kZPoints; ++t) {
      dT[t] += weight * phi[t];
    }
  }
#pragma unroll
  for (int t = 0; t < kZPoints; ++t) {
    temperature[p0 + t] = t_init + dT[t];
  }
}

// Depths past the last full group of four. One thread visits one leftover point.
__global__ void dense_temperature_ztile_tail(const double* __restrict__ xp, const double* __restrict__ yp,
                                             const double* __restrict__ zp, const double* __restrict__ xb,
                                             const double* __restrict__ yb, const double* __restrict__ zb,
                                             const double* __restrict__ phix, const double* __restrict__ phiy,
                                             const double* __restrict__ phiz, const double* __restrict__ dtau,
                                             const double* __restrict__ expmod, int npoints, int nnodes, int nz,
                                             double t_init, double* __restrict__ temperature) {
  const int nrem = nz % kZPoints;
  const int nrows = npoints / nz;
  const int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= nrows * nrem) {
    return;
  }
  const int row = id / nrem;
  const int k = nz - nrem + (id - row * nrem);
  const int p = row * nz + k;
  const double x = xp[p];
  const double y = yp[p];
  const double z = zp[p];
  double dT = 0.0;
  int node = 0;
  for (; node + 1 < nnodes; node += 2) {
    dT += snapshot_heat(x, y, z, xb[node], yb[node], zb[node], phix[node], phiy[node], phiz[node], dtau[node],
                        expmod[node]);
    dT += snapshot_heat(x, y, z, xb[node + 1], yb[node + 1], zb[node + 1], phix[node + 1], phiy[node + 1],
                        phiz[node + 1], dtau[node + 1], expmod[node + 1]);
  }
  if (node < nnodes) {
    dT += snapshot_heat(x, y, z, xb[node], yb[node], zb[node], phix[node], phiy[node], phiz[node], dtau[node],
                        expmod[node]);
  }
  temperature[p] = t_init + dT;
}

// Allocate device memory and copy one host array into it.
double* upload(const std::vector<double>& host) {
  double* device = nullptr;
  CUDA_CHECK(cudaMalloc(&device, host.size() * sizeof(double)));
  CUDA_CHECK(cudaMemcpy(device, host.data(), host.size() * sizeof(double), cudaMemcpyHostToDevice));
  return device;
}

// Check each device field against the host, then report the average time of each
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

  double* xp = upload(grid.xp);
  double* yp = upload(grid.yp);
  double* zp = upload(grid.zp);
  double* xb = upload(nodes.xb);
  double* yb = upload(nodes.yb);
  double* zb = upload(nodes.zb);
  double* phix = upload(nodes.phix);
  double* phiy = upload(nodes.phiy);
  double* phiz = upload(nodes.phiz);
  double* dtau = upload(nodes.dtau);
  double* expmod = upload(nodes.expmod);
  double* temperature = nullptr;
  CUDA_CHECK(cudaMalloc(&temperature, static_cast<size_t>(npoints) * sizeof(double)));

  const int warps_per_block = block / 32;
  const int warp_grid = (npoints + warps_per_block - 1) / warps_per_block;
  const int point_grid = (npoints + block - 1) / block;
  const int z_rows = nx * ny;
  const int ztile_work = z_rows * (nz / kZPoints);
  const int tail_work = z_rows * (nz % kZPoints);
  const int ztile_grid = (ztile_work + block - 1) / block;
  const int tail_grid = (tail_work + block - 1) / block;
  auto launch_warp = [&]() {
    dense_temperature_team<<<warp_grid, block>>>(xp, yp, zp, xb, yb, zb, phix, phiy, phiz, dtau,
                                                 expmod, npoints, nnodes, kTInit, temperature);
  };
  auto launch_point = [&]() {
    dense_temperature_point<<<point_grid, block>>>(xp, yp, zp, xb, yb, zb, phix, phiy, phiz, dtau,
                                                   expmod, npoints, nnodes, kTInit, temperature);
  };
  auto launch_ztile = [&]() {
    if (ztile_work > 0) {
      dense_temperature_ztile<<<ztile_grid, block>>>(xp, yp, zp, xb, yb, zb, phix, phiy, phiz, dtau,
                                                     expmod, npoints, nnodes, nz, kTInit, temperature);
    }
    if (tail_work > 0) {
      dense_temperature_ztile_tail<<<tail_grid, block>>>(xp, yp, zp, xb, yb, zb, phix, phiy, phiz, dtau,
                                                         expmod, npoints, nnodes, nz, kTInit, temperature);
    }
  };

  std::vector<double> host_T(static_cast<size_t>(npoints));
  auto check = [&](auto&& launch_kernel, double& max_abs, double& max_rel) {
    launch_kernel();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(host_T.data(), temperature, host_T.size() * sizeof(double),
                          cudaMemcpyDeviceToHost));
    return compare_temperature(host_T, nodes, grid, kTInit, max_abs, max_rel);
  };
  auto average_ms = [&](auto&& launch_kernel) {
    for (int i = 0; i < warmup; ++i) {
      launch_kernel();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
      launch_kernel();
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count() / static_cast<double>(iters);
  };

  double warp_abs = 0.0;
  double warp_rel = 0.0;
  double point_abs = 0.0;
  double point_rel = 0.0;
  double ztile_abs = 0.0;
  double ztile_rel = 0.0;
  const bool warp_pass = check(launch_warp, warp_abs, warp_rel);
  const bool point_pass = check(launch_point, point_abs, point_rel);
  const bool ztile_pass = check(launch_ztile, ztile_abs, ztile_rel);

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("device: %s  sm_%d%d  compute units: %d\n", prop.name, prop.major, prop.minor,
         prop.multiProcessorCount);
  printf("grid: %d x %d x %d (%d points)  nodes: %d  block: %d\n", nx, ny, nz, npoints, nnodes,
         block);
  printf("warmup: %d  iters: %d\n", warmup, iters);
  printf("warp max abs error: %.3e  max rel error: %.3e\n", warp_abs, warp_rel);
  printf("warp-agnostic max abs error: %.3e  max rel error: %.3e\n", point_abs, point_rel);
  printf("z-tile max abs error: %.3e  max rel error: %.3e\n", ztile_abs, ztile_rel);

  int status = EXIT_SUCCESS;
  if (!warp_pass || !point_pass || !ztile_pass) {
    printf("FAIL\n");
    status = EXIT_FAILURE;
  } else {
    printf("Average kernel time (warp): %.4f ms\n", average_ms(launch_warp));
    printf("Average kernel time (warp-agnostic): %.4f ms\n", average_ms(launch_point));
    printf("Average kernel time (z-tile): %.4f ms\n", average_ms(launch_ztile));
    printf("PASS\n");
  }

  cudaFree(xp);
  cudaFree(yp);
  cudaFree(zp);
  cudaFree(xb);
  cudaFree(yb);
  cudaFree(zb);
  cudaFree(phix);
  cudaFree(phiy);
  cudaFree(phiz);
  cudaFree(dtau);
  cudaFree(expmod);
  cudaFree(temperature);
  return status;
}

// Parse the grid size, beam-snapshot count, and launch settings, then run.
int main(int argc, char** argv) {
  const BenchConfig cfg = parse_options(argc, argv);
  if (cfg.block < 32 || cfg.block > 1024 || (cfg.block % 32) != 0) {
    fprintf(stderr, "--block must be a multiple of 32 in [32, 1024]\n");
    return EXIT_FAILURE;
  }
  return run(cfg);
}
