// Host setup and the temperature check 

#pragma once

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "condor_reference.hpp"

constexpr double kAbsTol = 1e-8;
constexpr double kRelTol = 1e-9;
// exp(x) rounds to zero in binary64 below log(0.5 * denorm_min).
constexpr double kExpRoundsToZero = -745.13321910194110842;

#if defined(__CUDACC__) || defined(__HIPCC__)
#define CONDOR_FN __host__ __device__ __forceinline__
#else
#define CONDOR_FN inline
#endif

// Heat from one beam snapshot at one grid point. exp is evaluated only when its
// result is not the binary64 zero.
CONDOR_FN double snapshot_heat(double x, double y, double z, double xb, double yb, double zb,
                               double phix, double phiy, double phiz, double dtau, double expmod) {
  const double dx = x - xb;
  const double dy = y - yb;
  const double dz = z - zb;
  const double exponent =
      -3.0 * ((dx * dx * phix) + (dy * dy * phiy) + (dz * dz * phiz)) + expmod;
  const double phi = (exponent < kExpRoundsToZero) ? 0.0 : exp(exponent);
  return dtau * phi;
}
#undef CONDOR_FN

struct Nodes {
  std::vector<double> xb, yb, zb, phix, phiy, phiz, dtau, expmod;
};

struct Grid {
  int nx = 0;
  int ny = 0;
  int nz = 0;
  double xmin = 0.0;
  double xmax = 0.0;
  double ymin = 0.0;
  double ymax = 0.0;
  double zmin = 0.0;
  double zmax = 0.0;
  double xres = 0.0;
  double yres = 0.0;
  double zres = 0.0;
  std::vector<double> xp, yp, zp;
};

struct BenchConfig {
  int nx = 64;
  int ny = 64;
  int nz = 64;
  int nnodes = 512;
  int iters = 10;
  int warmup = 2;
  int block = 128;
};

// Flatten a grid location to Condor's linear index. k is contiguous and i is slowest:
// p = i*(ny*nz) + j*nz + k.
inline int ijk_to_p(int i, int j, int k, int ny, int nz) {
  return i * (ny * nz) + j * nz + k;
}

// Print the accepted command line.
inline void usage(const char* argv0) {
  fprintf(stderr,
          "usage: %s [--nx N] [--ny N] [--nz N] [--nodes N] [--iters N] [--warmup N] "
          "[--block N]\n",
          argv0);
}

// Read one non-negative integer option. Zero is rejected unless allow_zero is set.
inline int parse_int(const char* text, const char* name, bool allow_zero) {
  char* end = nullptr;
  errno = 0;
  const long value = std::strtol(text, &end, 10);
  if (end == text || *end != '\0' || errno == ERANGE || value < 0 ||
      value > static_cast<long>(INT_MAX) || (!allow_zero && value == 0)) {
    fprintf(stderr, "invalid %s: %s\n", name, text);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(value);
}

// Point indices are signed ints. Each axis and the point count stay at or below
// 2e9, leaving room for npoints + block - 1 when the block size is at most 1024.
constexpr int kMaxPoints = 2000000000;

// True when an axis or the point count exceeds kMaxPoints. The divisions reject
// a product that does not fit in a signed int without multiplying the axes first.
inline bool grid_exceeds_points(int nx, int ny, int nz) {
  if (nx > kMaxPoints || ny > kMaxPoints || nz > kMaxPoints || ny > kMaxPoints / nx) {
    return true;
  }
  const int xy = nx * ny;
  return nz > kMaxPoints / xy;
}

// Read the grid size, beam-snapshot count, and launch settings.
inline BenchConfig parse_options(int argc, char** argv) {
  BenchConfig cfg;
  for (int i = 1; i < argc; ++i) {
    auto need = [&](const char* flag) {
      if (i + 1 >= argc) {
        fprintf(stderr, "missing value for %s\n", flag);
        usage(argv[0]);
        std::exit(EXIT_FAILURE);
      }
      return argv[++i];
    };
    if (std::strcmp(argv[i], "--nx") == 0) {
      cfg.nx = parse_int(need("--nx"), "--nx", false);
    } else if (std::strcmp(argv[i], "--ny") == 0) {
      cfg.ny = parse_int(need("--ny"), "--ny", false);
    } else if (std::strcmp(argv[i], "--nz") == 0) {
      cfg.nz = parse_int(need("--nz"), "--nz", false);
    } else if (std::strcmp(argv[i], "--nodes") == 0) {
      cfg.nnodes = parse_int(need("--nodes"), "--nodes", false);
    } else if (std::strcmp(argv[i], "--iters") == 0) {
      cfg.iters = parse_int(need("--iters"), "--iters", false);
    } else if (std::strcmp(argv[i], "--warmup") == 0) {
      cfg.warmup = parse_int(need("--warmup"), "--warmup", true);
    } else if (std::strcmp(argv[i], "--block") == 0) {
      cfg.block = parse_int(need("--block"), "--block", false);
    } else if (std::strcmp(argv[i], "--help") == 0) {
      usage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else {
      fprintf(stderr, "unknown argument: %s\n", argv[i]);
      usage(argv[0]);
      std::exit(EXIT_FAILURE);
    }
  }
  if (grid_exceeds_points(cfg.nx, cfg.ny, cfg.nz)) {
    fprintf(stderr, "grid is too large\n");
    std::exit(EXIT_FAILURE);
  }
  return cfg;
}

// Build the beam snapshots. They sample one straight 2 mm pass lasting 4 ms.
// Positions and time-step weights are synthetic. The ellipsoid widths and the
// exponential modifier use Condor's moving-line expressions from GaussIntegrate:
// phix = 1/(ax^2 + 12*diffusivity*tau) and expmod = 0.5*log(qmod^2*phix*phiy*phiz).
// qmod is the heat coefficient and does not depend on the widths.
inline void build_scan(Nodes& nodes, int nnodes) {
  constexpr double kPi = 3.14159265358979323846;
  constexpr double kXmin = 0.0;
  constexpr double kXmax = 2.0e-3;
  constexpr double kAx = 80.0e-6;
  constexpr double kAy = 60.0e-6;
  constexpr double kAz = 40.0e-6;
  constexpr double kDiffusivity = 5.0e-6;
  constexpr double kHistory = 4.0e-3;
  constexpr double kHeatScale = 2.0e7;

  nodes.xb.resize(static_cast<size_t>(nnodes));
  nodes.yb.resize(static_cast<size_t>(nnodes));
  nodes.zb.resize(static_cast<size_t>(nnodes));
  nodes.phix.resize(static_cast<size_t>(nnodes));
  nodes.phiy.resize(static_cast<size_t>(nnodes));
  nodes.phiz.resize(static_cast<size_t>(nnodes));
  nodes.dtau.resize(static_cast<size_t>(nnodes));
  nodes.expmod.resize(static_cast<size_t>(nnodes));

  for (int node = 0; node < nnodes; ++node) {
    const double s = (nnodes == 1) ? 0.5 : static_cast<double>(node) / static_cast<double>(nnodes - 1);
    const size_t n = static_cast<size_t>(node);
    nodes.xb[n] = kXmin + s * (kXmax - kXmin);
    nodes.yb[n] = 20.0e-6 * std::sin(2.0 * kPi * s);
    nodes.zb[n] = 0.0;
    const double tau = (1.0 - s) * kHistory;
    const double ct = 12.0 * kDiffusivity * tau;
    nodes.phix[n] = 1.0 / (kAx * kAx + ct);
    nodes.phiy[n] = 1.0 / (kAy * kAy + ct);
    nodes.phiz[n] = 1.0 / (kAz * kAz + ct);
    nodes.dtau[n] = (kHistory / static_cast<double>(nnodes)) *
                    (0.85 + 0.15 * std::cos(8.0 * s));
    const double qmod = kHeatScale * (0.95 + 0.05 * std::sin(6.0 * s));
    const double phi_product = nodes.phix[n] * nodes.phiy[n] * nodes.phiz[n];
    nodes.expmod[n] = 0.5 * std::log(qmod * qmod * phi_product);
  }
}

// Build the places where temperature is stored. x runs along the 2 mm track,
// y across the 1 mm width, and z from 0.4 mm deep up to the surface. One point
// on an axis sits on the far face of that axis, with zero spacing.
inline void build_grid(Grid& grid, int nx, int ny, int nz) {
  constexpr double kXmin = 0.0;
  constexpr double kXmax = 2.0e-3;
  constexpr double kYmin = -0.5e-3;
  constexpr double kYmax = 0.5e-3;
  constexpr double kZmin = -0.4e-3;
  constexpr double kZmax = 0.0;

  grid.nx = nx;
  grid.ny = ny;
  grid.nz = nz;
  grid.xmin = (nx == 1) ? kXmax : kXmin;
  grid.xmax = kXmax;
  grid.ymin = (ny == 1) ? kYmax : kYmin;
  grid.ymax = kYmax;
  grid.zmin = (nz == 1) ? kZmax : kZmin;
  grid.zmax = kZmax;
  grid.xres = (nx == 1) ? 0.0 : (kXmax - kXmin) / static_cast<double>(nx - 1);
  grid.yres = (ny == 1) ? 0.0 : (kYmax - kYmin) / static_cast<double>(ny - 1);
  grid.zres = (nz == 1) ? 0.0 : (kZmax - kZmin) / static_cast<double>(nz - 1);

  const int npoints = nx * ny * nz;
  grid.xp.resize(static_cast<size_t>(npoints));
  grid.yp.resize(static_cast<size_t>(npoints));
  grid.zp.resize(static_cast<size_t>(npoints));
  for (int i = 0; i < nx; ++i) {
    for (int j = 0; j < ny; ++j) {
      for (int k = 0; k < nz; ++k) {
        const int p = ijk_to_p(i, j, k, ny, nz);
        grid.xp[static_cast<size_t>(p)] = grid.xmin + static_cast<double>(i) * grid.xres;
        grid.yp[static_cast<size_t>(p)] = grid.ymin + static_cast<double>(j) * grid.yres;
        grid.zp[static_cast<size_t>(p)] = grid.zmin + static_cast<double>(k) * grid.zres;
      }
    }
  }
}

// Compare the device temperature at every grid point with Condor's host reference.
// A point passes when the absolute error is at most 1e-8 or the relative error,
// scaled by max(|reference|, 1), is at most 1e-9.
inline bool compare_temperature(const std::vector<double>& device_T, const Nodes& nodes, const Grid& grid,
                                double t_init, double& max_abs, double& max_rel) {
  const int npoints = grid.nx * grid.ny * grid.nz;
  max_abs = 0.0;
  max_rel = 0.0;
  int failed = 0;

#ifdef _OPENMP
#pragma omp parallel for reduction(max : max_abs, max_rel) reduction(max : failed)
#endif
  for (int p = 0; p < npoints; ++p) {
    const double reference = condor_reference::temperature(nodes, grid, t_init, p);
    const double got = device_T[static_cast<size_t>(p)];
    const double abs_err = std::fabs(got - reference);
    const double scale = std::max(std::fabs(reference), 1.0);
    const double rel_err = abs_err / scale;
    max_abs = std::max(max_abs, abs_err);
    max_rel = std::max(max_rel, rel_err);
    if (!(abs_err <= kAbsTol || rel_err <= kRelTol)) {
      failed = 1;
    }
  }
  return failed == 0;
}
