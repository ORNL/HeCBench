#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <sycl/sycl.hpp>

#include "condor_host.hpp"

// Read-only global load. On NVIDIA this is ld.global.nc, the same cache path
// nvcc uses for the const node arrays. Other targets use an ordinary load.
__attribute__((always_inline)) inline double load(const double* p) {
#if defined(__NVPTX__)
  return __nvvm_ldg_d(p);
#else
  return *p;
#endif
}

__attribute__((always_inline)) inline double snapshot_at(double x, double y, double z, const double* xb,
                                                         const double* yb, const double* zb, const double* phix,
                                                         const double* phiy, const double* phiz, const double* dtau,
                                                         const double* expmod, int node) {
  return snapshot_heat(x, y, z, load(xb + node), load(yb + node), load(zb + node), load(phix + node),
                       load(phiy + node), load(phiz + node), load(dtau + node), load(expmod + node));
}

template <int Wave>
double warp_sum(sycl::sub_group sg, double value) {
#pragma unroll
  for (int offset = Wave / 2; offset > 0; offset >>= 1) {
    value += sycl::shift_group_left(sg, value, static_cast<uint32_t>(offset));
  }
  return value;
}

// Temperature at every grid point. One sub-group owns one point and splits the beam
// snapshots across its lanes. Lane 0 writes T_init plus the summed contributions.
// Several sub-groups share a work-group so a sub-group does not need a work-group
// of its own. Every lane in the sub-group has the same point index, so the return
// and the sub-group reduction stay converged. Wave is the compile-time sub-group
// size: the stride has to be constant for the two independent exp evaluations in
// each iteration to overlap.
template <int Wave>
void dense_temperature_team(sycl::nd_item<1> &item, const double* __restrict__ xp,
                            const double* __restrict__ yp, const double* __restrict__ zp,
                            const double* __restrict__ xb, const double* __restrict__ yb,
                            const double* __restrict__ zb, const double* __restrict__ phix,
                            const double* __restrict__ phiy, const double* __restrict__ phiz,
                            const double* __restrict__ dtau, const double* __restrict__ expmod, int npoints,
                            int nnodes, double t_init, double* __restrict__ temperature) {
  const int local = item.get_local_id(0);
  const int lane = local & (Wave - 1);
  const int subs = static_cast<int>(item.get_local_range(0)) / Wave;
  const int p = static_cast<int>(item.get_group(0)) * subs + local / Wave;
  if (p >= npoints) {
    return;
  }
  const double x = load(xp + p);
  const double y = load(yp + p);
  const double z = load(zp + p);

  double dT = 0.0;
  int node = lane;
  for (; node + Wave < nnodes; node += Wave * 2) {
    dT += snapshot_at(x, y, z, xb, yb, zb, phix, phiy, phiz, dtau, expmod, node);
    dT += snapshot_at(x, y, z, xb, yb, zb, phix, phiy, phiz, dtau, expmod, node + Wave);
  }
  if (node < nnodes) {
    dT += snapshot_at(x, y, z, xb, yb, zb, phix, phiy, phiz, dtau, expmod, node);
  }
  dT = warp_sum<Wave>(item.get_sub_group(), dT);
  if (lane == 0) {
    temperature[p] = t_init + dT;
  }
}

// Same temperature field, with one work-item per grid point. The work-item visits
// the beam snapshots in order and uses no sub-group size or sub-group reduction,
// so the mapping is valid for any sub-group width.
void dense_temperature_point(sycl::nd_item<1> &item, const double* __restrict__ xp,
                             const double* __restrict__ yp, const double* __restrict__ zp,
                             const double* __restrict__ xb, const double* __restrict__ yb,
                             const double* __restrict__ zb, const double* __restrict__ phix,
                             const double* __restrict__ phiy, const double* __restrict__ phiz,
                             const double* __restrict__ dtau, const double* __restrict__ expmod, int npoints,
                             int nnodes, double t_init, double* __restrict__ temperature) {
  const int p = item.get_global_id(0);
  if (p >= npoints) {
    return;
  }
  const double x = load(xp + p);
  const double y = load(yp + p);
  const double z = load(zp + p);
  double dT = 0.0;
  int node = 0;
  // Two snapshots per iteration. Their exp evaluations are independent.
  for (; node + 1 < nnodes; node += 2) {
    dT += snapshot_at(x, y, z, xb, yb, zb, phix, phiy, phiz, dtau, expmod, node);
    dT += snapshot_at(x, y, z, xb, yb, zb, phix, phiy, phiz, dtau, expmod, node + 1);
  }
  if (node < nnodes) {
    dT += snapshot_at(x, y, z, xb, yb, zb, phix, phiy, phiz, dtau, expmod, node);
  }
  temperature[p] = t_init + dT;
}

// Points along z that one work-item keeps together. They share x and y.
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
#pragma unroll
  for (int t = 0; t < kZPoints; ++t) {
    const double dz = z[t] - zb;
    const double exponent = -3.0 * (xy + (dz * dz * phiz)) + expmod;
    phi[t] = (exponent < kExpRoundsToZero) ? 0.0 : exp(exponent);
  }
}

// Same temperature field. Each work-item owns four consecutive z points and applies
// two beam snapshots at a time.
void dense_temperature_ztile(sycl::nd_item<1>& item, const double* __restrict__ xp,
                             const double* __restrict__ yp, const double* __restrict__ zp,
                             const double* __restrict__ xb, const double* __restrict__ yb,
                             const double* __restrict__ zb, const double* __restrict__ phix,
                             const double* __restrict__ phiy, const double* __restrict__ phiz,
                             const double* __restrict__ dtau, const double* __restrict__ expmod, int npoints,
                             int nnodes, int nz, double t_init, double* __restrict__ temperature) {
  const int ntiles = nz / kZPoints;
  const int nrows = npoints / nz;
  const int id = item.get_global_id(0);
  if (id >= nrows * ntiles) {
    return;
  }
  const int row = id / ntiles;
  const int tile = id - row * ntiles;
  const int p0 = row * nz + tile * kZPoints;
  const double x = load(xp + p0);
  const double y = load(yp + p0);
  double z[kZPoints];
  double dT[kZPoints];
#pragma unroll
  for (int t = 0; t < kZPoints; ++t) {
    z[t] = load(zp + p0 + t);
    dT[t] = 0.0;
  }
  int node = 0;
  for (; node + 1 < nnodes; node += 2) {
    double phi0[kZPoints];
    double phi1[kZPoints];
    snapshot_phi_z(x, y, z, phi0, load(xb + node), load(yb + node), load(zb + node), load(phix + node),
                   load(phiy + node), load(phiz + node), load(expmod + node));
    snapshot_phi_z(x, y, z, phi1, load(xb + node + 1), load(yb + node + 1), load(zb + node + 1),
                   load(phix + node + 1), load(phiy + node + 1), load(phiz + node + 1),
                   load(expmod + node + 1));
    const double dtau0 = load(dtau + node);
    const double dtau1 = load(dtau + node + 1);
#pragma unroll
    for (int t = 0; t < kZPoints; ++t) {
      dT[t] += dtau0 * phi0[t];
      dT[t] += dtau1 * phi1[t];
    }
  }
  if (node < nnodes) {
    double phi[kZPoints];
    snapshot_phi_z(x, y, z, phi, load(xb + node), load(yb + node), load(zb + node), load(phix + node),
                   load(phiy + node), load(phiz + node), load(expmod + node));
    const double weight = load(dtau + node);
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

// Depths past the last full group of four. One work-item visits one leftover point.
void dense_temperature_ztile_tail(sycl::nd_item<1>& item, const double* __restrict__ xp,
                                  const double* __restrict__ yp, const double* __restrict__ zp,
                                  const double* __restrict__ xb, const double* __restrict__ yb,
                                  const double* __restrict__ zb, const double* __restrict__ phix,
                                  const double* __restrict__ phiy, const double* __restrict__ phiz,
                                  const double* __restrict__ dtau, const double* __restrict__ expmod, int npoints,
                                  int nnodes, int nz, double t_init, double* __restrict__ temperature) {
  const int nrem = nz % kZPoints;
  const int nrows = npoints / nz;
  const int id = item.get_global_id(0);
  if (id >= nrows * nrem) {
    return;
  }
  const int row = id / nrem;
  const int k = nz - nrem + (id - row * nrem);
  const int p = row * nz + k;
  const double x = load(xp + p);
  const double y = load(yp + p);
  const double z = load(zp + p);
  double dT = 0.0;
  int node = 0;
  for (; node + 1 < nnodes; node += 2) {
    dT += snapshot_at(x, y, z, xb, yb, zb, phix, phiy, phiz, dtau, expmod, node);
    dT += snapshot_at(x, y, z, xb, yb, zb, phix, phiy, phiz, dtau, expmod, node + 1);
  }
  if (node < nnodes) {
    dT += snapshot_at(x, y, z, xb, yb, zb, phix, phiy, phiz, dtau, expmod, node);
  }
  temperature[p] = t_init + dT;
}

// Allocate device memory and copy one host array into it.
double* upload(sycl::queue& q, const std::vector<double>& host) {
  double* device = sycl::malloc_device<double>(host.size(), q);
  if (device == nullptr) {
    fprintf(stderr, "SYCL error: device allocation failed\n");
    std::exit(EXIT_FAILURE);
  }
  q.memcpy(device, host.data(), host.size() * sizeof(double)).wait_and_throw();
  return device;
}

int max_sub_group_size(const sycl::device& device) {
  auto sg_sizes = device.get_info<sycl::info::device::sub_group_sizes>();
  auto r = std::max_element(sg_sizes.begin(), sg_sizes.end());
  return *r;
}

template <int Wave>
void launch_team(sycl::queue& q, int groups, int block, const double* xp, const double* yp, const double* zp,
                 const double* xb, const double* yb, const double* zb, const double* phix, const double* phiy,
                 const double* phiz, const double* dtau, const double* expmod, int npoints, int nnodes,
                 double t_init, double* temperature) {
  q.parallel_for(sycl::nd_range<1>(static_cast<size_t>(groups) * block, block),
                 [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(Wave)]] {
                   dense_temperature_team<Wave>(item, xp, yp, zp, xb, yb, zb, phix, phiy, phiz, dtau,
                                                expmod, npoints, nnodes, t_init, temperature);
                 });
}

// Launch the cooperative kernel at the device's maximum sub-group size.
void launch_team_wave(sycl::queue& q, int wave, int groups, int block, const double* xp, const double* yp,
                      const double* zp, const double* xb, const double* yb, const double* zb, const double* phix,
                      const double* phiy, const double* phiz, const double* dtau, const double* expmod,
                      int npoints, int nnodes, double t_init, double* temperature) {
  switch (wave) {
    case 8:
      launch_team<8>(q, groups, block, xp, yp, zp, xb, yb, zb, phix, phiy, phiz, dtau, expmod, npoints,
                     nnodes, t_init, temperature);
      break;
    case 16:
      launch_team<16>(q, groups, block, xp, yp, zp, xb, yb, zb, phix, phiy, phiz, dtau, expmod, npoints,
                      nnodes, t_init, temperature);
      break;
    case 32:
      launch_team<32>(q, groups, block, xp, yp, zp, xb, yb, zb, phix, phiy, phiz, dtau, expmod, npoints,
                      nnodes, t_init, temperature);
      break;
    case 64:
      launch_team<64>(q, groups, block, xp, yp, zp, xb, yb, zb, phix, phiy, phiz, dtau, expmod, npoints,
                      nnodes, t_init, temperature);
      break;
    default:
      fprintf(stderr, "unsupported sub-group size %d\n", wave);
      std::exit(EXIT_FAILURE);
  }
}

// Check each device field against the host, then report the average time of each
// kernel. wave is the sub-group size packed into the work-group.
int run(sycl::queue& q, const BenchConfig& cfg, int wave) {
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

  double* xp = upload(q, grid.xp);
  double* yp = upload(q, grid.yp);
  double* zp = upload(q, grid.zp);
  double* xb = upload(q, nodes.xb);
  double* yb = upload(q, nodes.yb);
  double* zb = upload(q, nodes.zb);
  double* phix = upload(q, nodes.phix);
  double* phiy = upload(q, nodes.phiy);
  double* phiz = upload(q, nodes.phiz);
  double* dtau = upload(q, nodes.dtau);
  double* expmod = upload(q, nodes.expmod);
  double* temperature = sycl::malloc_device<double>(static_cast<size_t>(npoints), q);
  if (temperature == nullptr) {
    fprintf(stderr, "SYCL error: device allocation failed\n");
    std::exit(EXIT_FAILURE);
  }

  const int waves_per_block = block / wave;
  const int warp_groups = (npoints + waves_per_block - 1) / waves_per_block;
  const int point_groups = (npoints + block - 1) / block;
  const int z_rows = nx * ny;
  const int ztile_work = z_rows * (nz / kZPoints);
  const int tail_work = z_rows * (nz % kZPoints);
  const int ztile_groups = (ztile_work + block - 1) / block;
  const int tail_groups = (tail_work + block - 1) / block;
  auto launch_warp = [&]() {
    launch_team_wave(q, wave, warp_groups, block, xp, yp, zp, xb, yb, zb, phix, phiy, phiz, dtau, expmod,
                     npoints, nnodes, kTInit, temperature);
  };
  auto launch_point = [&]() {
    q.parallel_for(sycl::nd_range<1>(static_cast<size_t>(point_groups) * block, block),
                   [=](sycl::nd_item<1> item) {
                     dense_temperature_point(item, xp, yp, zp, xb, yb, zb, phix, phiy, phiz, dtau, expmod,
                                             npoints, nnodes, kTInit, temperature);
                   });
  };
  auto launch_ztile = [&]() {
    if (ztile_work > 0) {
      q.parallel_for(sycl::nd_range<1>(static_cast<size_t>(ztile_groups) * block, block),
                     [=](sycl::nd_item<1> item) {
                       dense_temperature_ztile(item, xp, yp, zp, xb, yb, zb, phix, phiy, phiz, dtau, expmod,
                                               npoints, nnodes, nz, kTInit, temperature);
                     });
    }
    if (tail_work > 0) {
      q.parallel_for(sycl::nd_range<1>(static_cast<size_t>(tail_groups) * block, block),
                     [=](sycl::nd_item<1> item) {
                       dense_temperature_ztile_tail(item, xp, yp, zp, xb, yb, zb, phix, phiy, phiz, dtau,
                                                    expmod, npoints, nnodes, nz, kTInit, temperature);
                     });
    }
  };

  std::vector<double> host_T(static_cast<size_t>(npoints));
  auto check = [&](auto&& launch_kernel, double& max_abs, double& max_rel) {
    launch_kernel();
    q.wait_and_throw();
    q.memcpy(host_T.data(), temperature, host_T.size() * sizeof(double)).wait();
    return compare_temperature(host_T, nodes, grid, kTInit, max_abs, max_rel);
  };
  auto average_ms = [&](auto&& launch_kernel) {
    for (int i = 0; i < warmup; ++i) {
      launch_kernel();
    }
    q.wait();
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i) {
      launch_kernel();
    }
    q.wait();
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

  const sycl::device device = q.get_device();
  const std::string name = device.get_info<sycl::info::device::name>();
  const int units = static_cast<int>(device.get_info<sycl::info::device::max_compute_units>());
  printf("device: %s  wave: %d  compute units: %d\n", name.c_str(), wave, units);
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

  sycl::free(xp, q);
  sycl::free(yp, q);
  sycl::free(zp, q);
  sycl::free(xb, q);
  sycl::free(yb, q);
  sycl::free(zb, q);
  sycl::free(phix, q);
  sycl::free(phiy, q);
  sycl::free(phiz, q);
  sycl::free(dtau, q);
  sycl::free(expmod, q);
  sycl::free(temperature, q);
  return status;
}

// Parse the grid size, beam-snapshot count, and launch settings, then run.
int main(int argc, char** argv) {
  const BenchConfig cfg = parse_options(argc, argv);
  try {
#ifdef USE_GPU
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
    sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif
    const sycl::device device = q.get_device();
    const int wave = max_sub_group_size(device);
    if (wave != 8 && wave != 16 && wave != 32 && wave != 64) {
      fprintf(stderr, "unsupported sub-group size %d\n", wave);
      return EXIT_FAILURE;
    }
    const size_t max_group = device.get_info<sycl::info::device::max_work_group_size>();
    const int max_block = max_group < 1024 ? static_cast<int>(max_group) : 1024;
    if (cfg.block < wave || cfg.block > max_block || (cfg.block % wave) != 0) {
      fprintf(stderr, "--block must be a multiple of the sub-group size (%d) in [%d, %d]\n", wave,
              wave, max_block);
      return EXIT_FAILURE;
    }
    return run(q, cfg, wave);
  } catch (const sycl::exception& error) {
    fprintf(stderr, "SYCL error: %s\n", error.what());
    return EXIT_FAILURE;
  }
}
