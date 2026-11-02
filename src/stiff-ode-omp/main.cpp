// Isothermal constant-density backward Euler for the detailed n-heptane size
// (560 species, 2500 reactions), OpenMP target offload. Same program as
// stiff-ode-hip.
//
// Each step is a real stiff solve: analytic Jacobian, modified Newton, and a
// dense 560x560 LU with partial pivoting. One zone is one team. Temperature is
// fixed per zone, so kf, k0, and Kc are computed once.
//
// The kernel (full) runs split assembly, a sub-panel LU through a team-shared
// tile, and a blocked substitution. Its shape (team size, panel width, and
// tile width) is tuned for the wavefront width (SolverShapeFor). The driver
// checks one launch against the host reference, which integrates every zone
// over OpenMP threads on the host, and returns before the timed launches if
// that check fails. OpenMP has no occupancy API; the team-shared tile allows
// one team per CU, so the default zone count is the CU count.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

#include <omp.h>

#ifdef STIFF_ODE_HAS_HSA
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#endif

#ifdef STIFF_ODE_HAS_CUDA
#include <cuda_runtime.h>
#endif

#include "kinetics.h"
#include "../stiff-ode-cuda/workload.h"
#include "../stiff-ode-cuda/reference.h"

constexpr double kRelTol = 1e-12;
constexpr double kJacTol = 1e-4;

struct DeviceInfo {
  char name[128];
  char arch[160];
  int cus = 0;
  int wavefront = 0;
  int index = 0;
};

// Prints the accepted arguments.
void usage(const char* argv0) {
  fprintf(stderr,
          "usage: %s [--zones N] [--iters N] [--steps N] [--warmup N] [--dt SEC]\n",
          argv0);
}

#ifdef STIFF_ODE_HAS_HSA
// Copies src into dst, dropping leading and trailing spaces.
static void trim_copy(char* dst, size_t dstn, const char* src) {
  while (*src == ' ') ++src;
  snprintf(dst, dstn, "%s", src);
  size_t n = std::strlen(dst);
  while (n > 0 && dst[n - 1] == ' ') dst[--n] = '\0';
}

// atexit hook. HSA stays up until the process exits.
static void shutdown_hsa() { hsa_shut_down(); }

struct HsaPick {
  DeviceInfo* info;
  int gpu_ord;
  bool found;
};

// Records the first 32- or 64-wide GPU with at least 16 CUs.
static hsa_status_t pick_agent(hsa_agent_t agent, void* data) {
  HsaPick* pick = static_cast<HsaPick*>(data);
  hsa_device_type_t type;
  if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type) != HSA_STATUS_SUCCESS)
    return HSA_STATUS_SUCCESS;
  if (type != HSA_DEVICE_TYPE_GPU) return HSA_STATUS_SUCCESS;
  uint32_t wave = 0, cu = 0;
  hsa_agent_get_info(agent, HSA_AGENT_INFO_WAVEFRONT_SIZE, &wave);
  hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_COMPUTE_UNIT_COUNT, &cu);
  if (!pick->found && (wave == 32 || wave == 64) && (int)cu >= 16) {
    char prod[64] = {};
    hsa_agent_get_info(agent, (hsa_agent_info_t)HSA_AMD_AGENT_INFO_PRODUCT_NAME, prod);
    hsa_isa_t isa{};
    hsa_agent_get_info(agent, HSA_AGENT_INFO_ISA, &isa);
    char isa_name[256] = {};
    uint32_t len = 0;
    if (hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME_LENGTH, &len) == HSA_STATUS_SUCCESS &&
        len > 0 && len < sizeof(isa_name)) {
      hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, isa_name);
    }
    trim_copy(pick->info->name, sizeof(pick->info->name), prod[0] ? prod : "AMD GPU");
    const char* gfx = std::strstr(isa_name, "gfx");
    snprintf(pick->info->arch, sizeof(pick->info->arch), "%s", gfx ? gfx : isa_name);
    pick->info->cus = (int)cu;
    pick->info->wavefront = (int)wave;
    pick->info->index = pick->gpu_ord;
    pick->found = true;
  }
  ++pick->gpu_ord;
  return HSA_STATUS_SUCCESS;
}

// Fills info from the HSA runtime.
static bool query_hsa(DeviceInfo& info) {
  if (hsa_init() != HSA_STATUS_SUCCESS) return false;
  HsaPick pick{&info, 0, false};
  hsa_iterate_agents(pick_agent, &pick);
  if (!pick.found) {
    hsa_shut_down();
    return false;
  }
  std::atexit(shutdown_hsa);
  return true;
}
#endif

#ifdef STIFF_ODE_HAS_CUDA
// Fills info from CUDA device 0.
static bool query_cuda(DeviceInfo& info) {
  int n = 0;
  if (cudaGetDeviceCount(&n) != cudaSuccess || n < 1) return false;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return false;
  snprintf(info.name, sizeof(info.name), "%s", prop.name);
  snprintf(info.arch, sizeof(info.arch), "sm_%d%d", prop.major, prop.minor);
  info.cus = prop.multiProcessorCount;
  info.wavefront = prop.warpSize;
  info.index = 0;
  return true;
}
#endif

// Fills info from KFD topology sysfs.
static bool query_kfd(DeviceInfo& info) {
  int gpu_ord = 0;
  bool found = false;
  for (int node = 0; node < 64; ++node) {
    char path[256];
    snprintf(path, sizeof(path), "/sys/class/kfd/kfd/topology/nodes/%d/properties", node);
    FILE* f = std::fopen(path, "r");
    if (!f) continue;
    int simd_count = 0, simd_per_cu = 0, wave = 0, gfx = 0, device_id = 0, vendor = 0;
    char key[128];
    long val = 0;
    while (std::fscanf(f, "%127s %ld", key, &val) == 2) {
      if (!std::strcmp(key, "simd_count")) simd_count = (int)val;
      else if (!std::strcmp(key, "simd_per_cu")) simd_per_cu = (int)val;
      else if (!std::strcmp(key, "wave_front_size")) wave = (int)val;
      else if (!std::strcmp(key, "gfx_target_version")) gfx = (int)val;
      else if (!std::strcmp(key, "device_id")) device_id = (int)val;
      else if (!std::strcmp(key, "vendor_id")) vendor = (int)val;
    }
    std::fclose(f);
    if (vendor != 4098 || simd_count <= 0 || simd_per_cu <= 0) continue;
    int cus = simd_count / simd_per_cu;
    if (!found && (wave == 32 || wave == 64) && cus >= 16) {
      info.cus = cus;
      info.wavefront = wave;
      info.index = gpu_ord;
      if (device_id == 29856 && gfx == 90402) {
        // hipDeviceProp_t for this SKU: name and gcnArchName.
        snprintf(info.name, sizeof(info.name), "AMD Instinct MI300A");
        snprintf(info.arch, sizeof(info.arch), "gfx942:sramecc+:xnack-");
      } else {
        int major = gfx / 10000;
        int minor = (gfx / 100) % 100;
        int step = gfx % 100;
        snprintf(info.name, sizeof(info.name), "AMD GPU %d", device_id);
        if (step < 10)
          snprintf(info.arch, sizeof(info.arch), "gfx%d%d%d", major, minor, step);
        else
          snprintf(info.arch, sizeof(info.arch), "gfx%d%d%x", major, minor, step);
      }
      found = true;
    }
    ++gpu_ord;
  }
  return found;
}

// OpenMP reports a device count, not a wavefront or CU count. The 32-wide
// shape is the one this port runs when the vendor query is absent.
static bool query_openmp(DeviceInfo& info) {
  const int ndev = omp_get_num_devices();
  if (ndev < 1) return false;
  const int dev = omp_get_default_device();
  snprintf(info.name, sizeof(info.name), "OpenMP device %d", dev);
  snprintf(info.arch, sizeof(info.arch), "spir64");
  info.cus = 0;
  info.wavefront = 32;
  info.index = dev;
  return true;
}

// Tries HSA, then CUDA, then KFD, then the OpenMP default device.
static bool query_device(DeviceInfo& info) {
  info = DeviceInfo{};
#ifdef STIFF_ODE_HAS_HSA
  if (query_hsa(info)) return true;
#endif
#ifdef STIFF_ODE_HAS_CUDA
  if (query_cuda(info)) return true;
#endif
  if (query_kfd(info)) return true;
  return query_openmp(info);
}

// Device buffers, the arguments of the HIP solver_kernel launch.
struct DeviceArgs {
  Tables tab;
  const int* row_off;
  const int* row_reac;
  const double* T;
  double* C;
  double* C0;
  double* w;
  double* rhs;
  double* J;
  int* perm;
  double* kf;
  double* k0;
  double* logKc;
  int* status;
  int* niters;
  double* part;
  int* diag;
  int zones;
};

#pragma omp declare target
// Every thread of a team's parallel region: the zones of that team, with the
// team-shared tile, pivot partials, and scalars. diag[0..1] records the team
// size and team count.
template <int Ws>
inline void team_zones(const DeviceArgs& a, int steps, double dt, double* tile, double* srel,
                       int* sfail, double* piv_v, int* piv_i) {
  const int nteams = omp_get_num_teams();
  const int team = omp_get_team_num();
  TeamCtx ctx;
  ctx.tid = omp_get_thread_num();
  ctx.bdim = omp_get_num_threads();
  ctx.tile = tile;
  ctx.srel = srel;
  ctx.sfail = sfail;
  ctx.piv_v = piv_v;
  ctx.piv_i = piv_i;
  if (team == 0 && ctx.tid == 0) {
    a.diag[0] = ctx.bdim;
    a.diag[1] = nteams;
  }
  for (int zone = team; zone < a.zones; zone += nteams) {
    ctx.zone = zone;
    solver_kernel<Ws>(a.tab, a.row_off, a.row_reac, a.T, a.C, a.C0, a.w, a.rhs, a.J, a.perm, a.kf,
                  a.k0, a.logKc, steps, dt, a.status, a.niters, a.part, ctx);
#pragma omp barrier
  }
}
#pragma omp end declare target

// One team per zone. The team-shared arrays are the HIP dynamic LDS and the
// __shared__ scalars. Clang takes them as statics in omp_pteam_mem_alloc.
// icpx ignores that allocator on a static and gives each thread its own copy,
// so the pivot tile is not shared and a later team barrier does not complete.
// omp_alloc(omp_pteam_mem_alloc) in the teams region is the shared tile on
// icpx. NVHPC has no allocate directive; it puts variables declared in the
// teams region, outside the parallel region, in CUDA shared memory when they
// fit in 48 KB (Makefile.nvc sizes the tile for that).
template <int Ws>
void launch_solver(const DeviceArgs& a, int steps, double dt) {
  constexpr int kTileWords = kTileBytes<SolverShapeFor<Ws>::sub> / 8;
  const int zones = a.zones;
  const DeviceArgs args = a;
#if defined(__INTEL_LLVM_COMPILER)
#pragma omp target teams num_teams(zones) thread_limit(SolverShapeFor<Ws>::block) firstprivate(args)
  {
    double* team_tile =
        static_cast<double*>(omp_alloc(sizeof(double) * kTileWords, omp_pteam_mem_alloc));
    double* team_srel = static_cast<double*>(omp_alloc(sizeof(double), omp_pteam_mem_alloc));
    int* team_sfail = static_cast<int*>(omp_alloc(sizeof(int), omp_pteam_mem_alloc));
    double* team_piv_v =
        static_cast<double*>(omp_alloc(sizeof(double) * kMaxWave, omp_pteam_mem_alloc));
    int* team_piv_i = static_cast<int*>(omp_alloc(sizeof(int) * kMaxWave, omp_pteam_mem_alloc));
    if (team_tile && team_srel && team_sfail && team_piv_v && team_piv_i) {
#pragma omp parallel num_threads(SolverShapeFor<Ws>::block)
      team_zones<Ws>(args, steps, dt, team_tile, team_srel, team_sfail, team_piv_v, team_piv_i);
    } else if (omp_get_team_num() == 0) {
      args.diag[0] = -2;
    }
    omp_free(team_piv_i, omp_pteam_mem_alloc);
    omp_free(team_piv_v, omp_pteam_mem_alloc);
    omp_free(team_sfail, omp_pteam_mem_alloc);
    omp_free(team_srel, omp_pteam_mem_alloc);
    omp_free(team_tile, omp_pteam_mem_alloc);
  }
#elif defined(__NVCOMPILER)
#pragma omp target teams num_teams(zones) thread_limit(SolverShapeFor<Ws>::block) firstprivate(args)
  {
    double team_tile[kTileWords];
    double team_srel;
    int team_sfail;
    double team_piv_v[kMaxWave];
    int team_piv_i[kMaxWave];
#pragma omp parallel num_threads(SolverShapeFor<Ws>::block)
    team_zones<Ws>(args, steps, dt, team_tile, &team_srel, &team_sfail, team_piv_v, team_piv_i);
  }
#else
#pragma omp target teams num_teams(zones) thread_limit(SolverShapeFor<Ws>::block) firstprivate(args)
#pragma omp parallel num_threads(SolverShapeFor<Ws>::block)
  {
    static double team_tile[kTileWords];
    static double team_srel;
    static int team_sfail;
    static double team_piv_v[kMaxWave];
    static int team_piv_i[kMaxWave];
#pragma omp allocate(team_tile, team_srel, team_sfail, team_piv_v, team_piv_i) \
    allocator(omp_pteam_mem_alloc)
    team_zones<Ws>(args, steps, dt, team_tile, &team_srel, &team_sfail, team_piv_v, team_piv_i);
  }
#endif
}

// Verifies one launch against the host reference, then times the kernel.
int run(int zones, int iters, int steps, int warmup, double dt_user, const DeviceInfo& prop) {
  if (zones <= 0) zones = prop.cus;
  if (zones <= 0) {
    fprintf(stderr, "OpenMP did not report a compute-unit count; pass --zones N\n");
    return EXIT_FAILURE;
  }
  SynthMech synth;
  build_synthetic(synth);
  Tables tab{};
  tab.rtype = synth.rtype.data();
  tab.reversible = synth.reversible.data();
  tab.nreact = synth.nreact.data();
  tab.nprod = synth.nprod.data();
  tab.r_idx = synth.r_idx.data();
  tab.r_nu = synth.r_nu.data();
  tab.p_idx = synth.p_idx.data();
  tab.p_nu = synth.p_nu.data();
  tab.eff_off = synth.eff_off.data();
  tab.eff_len = synth.eff_len.data();
  tab.eff_sp = synth.eff_sp.data();
  tab.eff_eps = synth.eff_eps.data();
  tab.troe = synth.troe.data();
  tab.A_high = synth.A_high.data();
  tab.B_high = synth.B_high.data();
  tab.Ea_high = synth.Ea_high.data();
  tab.A_low = synth.A_low.data();
  tab.B_low = synth.B_low.data();
  tab.Ea_low = synth.Ea_low.data();
  tab.nasa_lo = synth.nasa_lo.data();
  tab.nasa_hi = synth.nasa_hi.data();
  tab.tmid = synth.tmid.data();
  const int* comp = synth.comp.data();
  const SynthIds id = synth.ids;

  std::vector<int> row_off(NS + 1, 0);
  std::vector<int> row_reac;
  int max_row = 0;
  {
    std::vector<std::vector<int>> tmp(NS);
    for (int r = 0; r < NR; ++r) {
      for (int s = 0; s < NS; ++s) {
        if (net_nu(tab, r, s) != 0) tmp[s].push_back(r);
      }
    }
    for (int s = 0; s < NS; ++s) {
      row_off[s + 1] = row_off[s] + (int)tmp[s].size();
      max_row = std::max(max_row, (int)tmp[s].size());
      row_reac.insert(row_reac.end(), tmp[s].begin(), tmp[s].end());
    }
  }

  printf("device: %s  arch: %s  compute units: %d  wavefront: %d\n", prop.name, prop.arch, prop.cus,
         prop.wavefront);
  const SolverShape shape = solver_shape(prop.wavefront);
  printf("species: %d  reactions: %d  csr: %d  max_row: %d\n", NS, NR, (int)row_reac.size(),
         max_row);
  printf("block: %d  panel: %d  sub: %d  team_mem: %d\n", shape.block, shape.panel, shape.sub,
         shape.tile_bytes);
  printf("zones: %d  iters: %d  steps: %d  warmup: %d  newton_max: %d\n", zones, iters, steps,
         warmup, kNewtonMax);

  const int ncanon = 45;
  std::vector<double> Ccan((size_t)ncanon * NS);
  std::vector<double> Tcan(ncanon);
  for (int z = 0; z < ncanon; ++z)
    init_zone<NS>(z, Tcan[z], Ccan.data() + (size_t)z * NS, id.ch4, id.o2, id.n2, id.h, id.o,
                  id.oh, id.h2);

  std::vector<double> kf(NR), k0(NR), logKc(NR), w(NS), rhs(NS), J(NS * NS), Cwork(NS), C0(NS);
  std::vector<int> perm(NS);
  double lam = 0.0;
  int ilam = 0;
  double elem_w = 0.0;
  double jac_err = 0.0;
  for (int z = 0; z < ncanon; ++z) {
    prepare_all<NR>(tab, Tcan[z], kf.data(), k0.data(), logKc.data());
    assemble_all<NS>(tab, Ccan.data() + (size_t)z * NS, kf.data(), k0.data(), logKc.data(), Tcan[z],
                     row_off.data(), row_reac.data(), J.data(), w.data());
    double ez = element_residual<NS, NE>(w.data(), comp);
    if (z == 0 || ez > elem_w) elem_w = ez;
    for (int i = 0; i < NS; ++i) {
      double diag = -J[(size_t)i * NS + i];
      if (diag > lam) {
        lam = diag;
        ilam = z;
      }
    }
  }
  prepare_all<NR>(tab, Tcan[0], kf.data(), k0.data(), logKc.data());
  assemble_all<NS>(tab, Ccan.data(), kf.data(), k0.data(), logKc.data(), Tcan[0], row_off.data(),
                   row_reac.data(), J.data(), w.data());
  jac_err = jacobian_fd_error<NS, NR>(tab, Ccan.data(), kf.data(), k0.data(), logKc.data(), Tcan[0],
                                      row_off.data(), row_reac.data(), J.data());
  bool kinetics_ok = elem_w < 1e-9 && jac_err < kJacTol && std::isfinite(jac_err);
  printf("element_w: %.3e  jac_fd: %.3e  %s\n", elem_w, jac_err, kinetics_ok ? "ok" : "FAIL");
  if (!kinetics_ok) return EXIT_FAILURE;

  if (!(lam > 0.0)) {
    fprintf(stderr, "Jacobian diagonal is not stiff\n");
    return EXIT_FAILURE;
  }
  double dt = dt_user > 0.0 ? dt_user : 5.0 / lam;
  double dt_cfl = 5.0 / lam;
  // One host backward-Euler step of canonical zone z.
  auto try_step = [&](int z, double step_dt) {
    prepare_all<NR>(tab, Tcan[z], kf.data(), k0.data(), logKc.data());
    for (int i = 0; i < NS; ++i) Cwork[i] = Ccan[(size_t)z * NS + i];
    return host_step<NS, NR>(tab, Cwork.data(), Cwork.data(), C0.data(), w.data(), J.data(),
                             rhs.data(), perm.data(), kf.data(), k0.data(), logKc.data(), Tcan[z],
                             step_dt, row_off.data(), row_reac.data());
  };
  StepResult probe = try_step(ilam, dt);
  if (dt_user <= 0.0) {
    int shrinks = 0;
    while (probe.code != 0 && shrinks < 12) {
      dt *= 0.5;
      ++shrinks;
      probe = try_step(ilam, dt);
    }
  }
  double explicit_min = 1e300;
  double explicit_amp = 1.0;
  int istiff = 0;
  {
    prepare_all<NR>(tab, Tcan[ilam], kf.data(), k0.data(), logKc.data());
    assemble_all<NS>(tab, Ccan.data() + (size_t)ilam * NS, kf.data(), k0.data(), logKc.data(),
                     Tcan[ilam], row_off.data(), row_reac.data(), J.data(), w.data());
    for (int i = 0; i < NS; ++i) {
      explicit_min = std::min(explicit_min, Ccan[(size_t)ilam * NS + i] + dt * w[i]);
      if (-J[(size_t)i * NS + i] >= lam * (1.0 - 1e-12)) istiff = i;
    }
    explicit_amp = 1.0 + dt_cfl * J[(size_t)istiff * NS + istiff];
  }
  // One Euler step from partial equilibrium can stay non-negative. Doubling the
  // stiffest species and marching explicit Euler makes the CFL>1 mode go negative.
  double explicit_march = 0.0;
  {
    std::vector<double> Ce(Ccan.begin() + (size_t)ilam * NS, Ccan.begin() + (size_t)(ilam + 1) * NS);
    Ce[istiff] *= 2.0;
    prepare_all<NR>(tab, Tcan[ilam], kf.data(), k0.data(), logKc.data());
    for (int s = 0; s < 8; ++s) {
      assemble_all<NS>(tab, Ce.data(), kf.data(), k0.data(), logKc.data(), Tcan[ilam],
                       row_off.data(), row_reac.data(), J.data(), w.data());
      for (int i = 0; i < NS; ++i) {
        Ce[i] += dt_cfl * w[i];
        explicit_march = std::min(explicit_march, Ce[i]);
      }
      if (explicit_march < 0.0) break;
    }
  }
  printf("stiffness: %.6e  dt: %.6e  cfl: %.3f  newton_probe: %d iters %d rel %.3e\n", lam, dt,
         lam * dt, probe.code, probe.iters, probe.rel);
  printf("explicit_min: %.6e  explicit_amp: %.3f  explicit_march: %.6e\n", explicit_min,
         explicit_amp, explicit_march);
  if (probe.code != 0) {
    fprintf(stderr, "backward Euler did not converge during the dt search\n");
    return EXIT_FAILURE;
  }
  if (!(explicit_amp < 0.0) || !(explicit_march < 0.0)) {
    fprintf(stderr, "explicit Euler at CFL 5 did not show the stiff mode\n");
    return EXIT_FAILURE;
  }

  std::vector<double> state0((size_t)zones * NS);
  std::vector<double> temps(zones);
  for (int z = 0; z < zones; ++z)
    init_zone<NS>(z, temps[z], state0.data() + (size_t)z * NS, id.ch4, id.o2, id.n2, id.h, id.o,
                  id.oh, id.h2);

  const int dev = omp_get_default_device();
  const int host = omp_get_initial_device();
  std::vector<void*> allocs;
  bool alloc_ok = true;
  // Allocates n elements on the device.
  auto alloc = [&](auto*& p, size_t n) {
    p = (std::remove_reference_t<decltype(p)>)omp_target_alloc(sizeof(*p) * n, dev);
    if (!p) {
      fprintf(stderr, "omp_target_alloc of %zu bytes failed\n", sizeof(*p) * n);
      alloc_ok = false;
    } else {
      allocs.push_back(p);
    }
  };
  // Copies bytes from the host to the device.
  auto h2d = [&](void* d, const void* src, size_t bytes) {
    if (d && omp_target_memcpy(d, src, bytes, 0, 0, dev, host) != 0) {
      fprintf(stderr, "omp_target_memcpy to device failed\n");
      std::exit(EXIT_FAILURE);
    }
  };
  // Copies bytes from the device to the host.
  auto d2h = [&](void* dst, const void* d, size_t bytes) {
    if (omp_target_memcpy(dst, d, bytes, 0, 0, host, dev) != 0) {
      fprintf(stderr, "omp_target_memcpy to host failed\n");
      std::exit(EXIT_FAILURE);
    }
  };
  // Copies n ints to the device.
  auto copy_i = [&](const int* src, size_t n) {
    int* d;
    alloc(d, n);
    h2d(d, src, sizeof(int) * n);
    return d;
  };
  // Copies n doubles to the device.
  auto copy_d = [&](const double* src, size_t n) {
    double* d;
    alloc(d, n);
    h2d(d, src, sizeof(double) * n);
    return d;
  };

  Tables dtab{};
  dtab.rtype = copy_i(tab.rtype, NR);
  dtab.reversible = copy_i(tab.reversible, NR);
  dtab.nreact = copy_i(tab.nreact, NR);
  dtab.nprod = copy_i(tab.nprod, NR);
  dtab.r_idx = copy_i(tab.r_idx, NR * 3);
  dtab.r_nu = copy_i(tab.r_nu, NR * 3);
  dtab.p_idx = copy_i(tab.p_idx, NR * 3);
  dtab.p_nu = copy_i(tab.p_nu, NR * 3);
  dtab.eff_off = copy_i(tab.eff_off, NR);
  dtab.eff_len = copy_i(tab.eff_len, NR);
  dtab.eff_sp = copy_i(tab.eff_sp, N_EFF);
  dtab.eff_eps = copy_d(tab.eff_eps, N_EFF);
  dtab.troe = copy_d(tab.troe, NR * 4);
  dtab.A_high = copy_d(tab.A_high, NR);
  dtab.B_high = copy_d(tab.B_high, NR);
  dtab.Ea_high = copy_d(tab.Ea_high, NR);
  dtab.A_low = copy_d(tab.A_low, NR);
  dtab.B_low = copy_d(tab.B_low, NR);
  dtab.Ea_low = copy_d(tab.Ea_low, NR);
  dtab.nasa_lo = copy_d(tab.nasa_lo, NS * 7);
  dtab.nasa_hi = copy_d(tab.nasa_hi, NS * 7);
  dtab.tmid = copy_d(tab.tmid, NS);

  int* d_row_off = copy_i(row_off.data(), NS + 1);
  int* d_row_reac = copy_i(row_reac.data(), row_reac.size());
  double* d_T = copy_d(temps.data(), zones);
  double *d_C, *d_C0, *d_w, *d_rhs, *d_J, *d_kf, *d_k0, *d_logKc, *d_part;
  int *d_perm, *d_status, *d_niters;
  alloc(d_C, state0.size());
  alloc(d_C0, state0.size());
  alloc(d_w, state0.size());
  alloc(d_rhs, state0.size());
  alloc(d_J, state0.size() * NS);
  alloc(d_perm, state0.size());
  alloc(d_kf, (size_t)zones * NR);
  alloc(d_k0, (size_t)zones * NR);
  alloc(d_logKc, (size_t)zones * NR);
  alloc(d_status, zones);
  alloc(d_niters, zones);
  alloc(d_part, (size_t)zones * kRxTerms * NR);
  int* d_diag;
  alloc(d_diag, 2);

  // Frees the device buffers.
  auto cleanup = [&]() {
    for (void* p : allocs) omp_target_free(p, dev);
  };
  if (!alloc_ok) {
    cleanup();
    return EXIT_FAILURE;
  }

  DeviceArgs args{dtab, d_row_off, d_row_reac, d_T,     d_C,      d_C0,   d_w,
                  d_rhs, d_J,      d_perm,     d_kf,    d_k0,     d_logKc, d_status,
                  d_niters, d_part, d_diag,    zones};
  const int ws = prop.wavefront;
  // Launches one solver kernel for the device wavefront.
  auto launch = [&]() {
    if (ws == 64)
      launch_solver<64>(args, steps, dt);
    else
      launch_solver<32>(args, steps, dt);
  };

  struct Outcome {
    double seconds = 0;  // per timed launch
    int newton_bad = 0;
    int max_iters = 0;
    bool launched = true;
    std::vector<double> states;
  };

  const std::vector<int> zeros(zones, 0);
  // Untimed launches, then timed ones, all continuing from state0. The timed
  // interval is host wall time through the launches, which return after the
  // kernel. states holds the result after every launch has run. OpenMP may
  // field fewer threads than num_threads asks for (NVHPC sizes teams by
  // register use). The kernel strides by the team size, so any whole number of
  // waves gives the same answer; a partial wave fails. Fewer teams only stride
  // the zones.
  auto exec = [&](int untimed, int timed) {
    Outcome o;
    const int blk = shape.block;
    h2d(d_C, state0.data(), sizeof(double) * state0.size());
    h2d(d_status, zeros.data(), sizeof(int) * (size_t)zones);
    h2d(d_niters, zeros.data(), sizeof(int) * (size_t)zones);
    int diag0[2] = {-1, -1};
    h2d(d_diag, diag0, sizeof(diag0));
    for (int i = 0; i < untimed; ++i) launch();
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < timed; ++i) launch();
    const auto t1 = std::chrono::steady_clock::now();
    o.seconds = std::chrono::duration<double>(t1 - t0).count() / timed;
    int diag[2] = {0, 0};
    d2h(diag, d_diag, sizeof(diag));
    if (diag[0] == -2) {
      fprintf(stderr, "team allocation failed\n");
    } else if (diag[0] != blk) {
      printf("compiler_limit: requested %d threads, OpenMP launched %d\n", blk, diag[0]);
    }
    if (diag[0] != -2 && diag[1] != zones)
      printf("compiler_limit: requested %d teams, OpenMP launched %d (zones strided across teams)\n",
             zones, diag[1]);
    o.launched = diag[0] >= ws && diag[0] % ws == 0 && diag[1] >= 1;
    std::vector<int> st(zones, 0), ni(zones, 0);
    d2h(st.data(), d_status, sizeof(int) * (size_t)zones);
    d2h(ni.data(), d_niters, sizeof(int) * (size_t)zones);
    for (int z = 0; z < zones; ++z) {
      if (st[z] != 0) ++o.newton_bad;
      o.max_iters = std::max(o.max_iters, ni[z]);
    }
    o.states.resize(state0.size());
    d2h(o.states.data(), d_C, sizeof(double) * state0.size());
    if (o.newton_bad) printf("newton_fail full zones %d\n", o.newton_bad);
    return o;
  };

  // One launch from state0, checked against the host reference before warmup
  // and timing. The host reference runs after that check and finishes before
  // the timed launches, so its threads stay off the measurement.
  Outcome check = exec(0, 1);
  double drift = element_drift<NS, NE>(state0.data(), check.states.data(), comp);
  printf("element_drift: %.3e\n", drift);
  printf("zone0 T %.1f CH4 %.6e O2 %.6e H %.6e OH %.6e\n", temps[0], check.states[id.ch4],
         check.states[id.o2], check.states[id.h], check.states[id.oh]);
  printf("newton_iters_max: %d\n", check.max_iters);

  std::vector<double> ref = state0;
  auto t_cpu0 = std::chrono::steady_clock::now();
  const int cpu_bad =
      host_reference<NS, NR>(tab, temps, ref, steps, dt, row_off.data(), row_reac.data());
  const double cpu_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t_cpu0).count();
  // RESULT cpu <seconds> s <PASS|FAIL>
  //   seconds: wall time for the host reference to integrate every zone
  //   PASS: Newton succeeded in every zone
  printf("RESULT cpu %.6f s %s\n", cpu_seconds, cpu_bad ? "FAIL" : "PASS");
  if (cpu_bad) printf("newton_fail cpu zones %d\n", cpu_bad);
  Rel rel = compare_buf(ref, check.states);
  const bool check_ok = cpu_bad == 0 && check.launched && !check.newton_bad && rel.finite &&
                        rel.max_rel <= kRelTol;
  if (!check_ok) {
    printf("relative-error %.3e FAIL\n", rel.max_rel);
    printf("stiff-ode: FAIL\n");
    cleanup();
    return EXIT_FAILURE;
  }

  Outcome timed = exec(warmup, iters);
  const bool pass = timed.launched && !timed.newton_bad;
  // RESULT block-size <threads> <seconds> s relative-error <max-rel> <PASS|FAIL>
  //   block-size: threads requested per team
  //   seconds: mean host wall time of one timed launch (STEPS backward-Euler steps)
  //   relative-error: largest relative error of the check launch against the host reference
  //   PASS: the timed launch had a whole number of waves and Newton succeeded;
  //   the check was already within 1e-12
  printf("RESULT block-size %d %.6f s relative-error %.3e %s\n", shape.block, timed.seconds,
         rel.max_rel, pass ? "PASS" : "FAIL");
  printf("speedup_vs_cpu: %.3fx\n", cpu_seconds / timed.seconds);
  printf("stiff-ode: %s\n", pass ? "PASS" : "FAIL");
  cleanup();
  return pass ? 0 : EXIT_FAILURE;
}

// Parses the arguments, queries a device, and runs the benchmark.
int main(int argc, char** argv) {
  int zones = 0;
  int iters = 10;
  int steps = 100;
  int warmup = 5;
  double dt_user = 0.0;

  for (int i = 1; i < argc; ++i) {
    // Returns the argument that follows flag.
    auto need = [&](const char* flag) {
      if (i + 1 >= argc) {
        fprintf(stderr, "missing value for %s\n", flag);
        std::exit(EXIT_FAILURE);
      }
      return argv[++i];
    };
    if (!strcmp(argv[i], "--zones")) zones = std::atoi(need("--zones"));
    else if (!strcmp(argv[i], "--iters")) iters = std::atoi(need("--iters"));
    else if (!strcmp(argv[i], "--steps")) steps = std::atoi(need("--steps"));
    else if (!strcmp(argv[i], "--warmup")) warmup = std::atoi(need("--warmup"));
    else if (!strcmp(argv[i], "--dt")) dt_user = std::atof(need("--dt"));
    else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h")) {
      usage(argv[0]);
      return 0;
    } else {
      fprintf(stderr, "unknown argument %s\n", argv[i]);
      usage(argv[0]);
      return EXIT_FAILURE;
    }
  }
  if (iters < 1 || steps < 1 || warmup < 0) {
    fprintf(stderr, "iters and steps must be positive and warmup non-negative\n");
    return EXIT_FAILURE;
  }

  DeviceInfo prop;
  if (!query_device(prop)) {
    fprintf(stderr, "no device with a 32- or 64-wide wavefront\n");
    return EXIT_FAILURE;
  }
  if (omp_get_num_devices() < 1) {
    fprintf(stderr, "no OpenMP target device\n");
    return EXIT_FAILURE;
  }
  if (prop.index < 0 || prop.index >= omp_get_num_devices()) {
    fprintf(stderr, "OpenMP device count %d does not include selected GPU %d\n",
            omp_get_num_devices(), prop.index);
    return EXIT_FAILURE;
  }
  omp_set_default_device(prop.index);
  if (prop.wavefront != 32 && prop.wavefront != 64) {
    fprintf(stderr, "unsupported wavefront size %d\n", prop.wavefront);
    return EXIT_FAILURE;
  }

  return run(zones, iters, steps, warmup, dt_user, prop);
}
