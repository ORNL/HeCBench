// Isothermal constant-density backward Euler for the detailed n-heptane size (560 species, 2500 reactions)
//
// Each step is a real stiff solve: analytic Jacobian, modified Newton, and a
// dense 560x560 LU with partial pivoting. One zone is one block. Temperature
// is fixed per zone, so kf, k0, and Kc are computed once.
//
// The kernel (full) does split assembly, a panel LU through a shared-memory
// tile, and a blocked substitution. Its shape (block size, panel width, and
// tile width) is tuned for the wave width (solver_shape). The driver
// checks one launch against the host reference, which integrates every zone
// over OpenMP threads, and returns before the timed launches if that check
// fails. Default zone count is one per resident block, from the occupancy API.


#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "kinetics.cuh"
#include "../stiff-ode-cuda/workload.h"
#include "../stiff-ode-cuda/reference.h"

constexpr double kRelTol = 1e-12;
constexpr double kJacTol = 1e-4;

#define HIP_CHECK(expr)                                                        \
  do {                                                                         \
    hipError_t _err = (expr);                                                  \
    if (_err != hipSuccess) {                                                  \
      fprintf(stderr, "HIP error %s:%d %s\n", __FILE__, __LINE__,              \
              hipGetErrorString(_err));                                        \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// Prints the accepted arguments.
void usage(const char* argv0) {
  fprintf(stderr,
          "usage: %s [--zones N] [--iters N] [--steps N] [--warmup N] [--dt SEC]\n",
          argv0);
}

// Verifies one launch against the host reference, then times the kernel.
int run(int zones, int iters, int steps, int warmup, double dt_user, const hipDeviceProp_t& prop) {
  const SolverShape shape_s =
      prop.warpSize == 64 ? solver_shape<1024, 24, 12>() : solver_shape<512, 40, 8>();
  const SolverShape* shape = &shape_s;
  {
    hipFuncAttributes fa;
    HIP_CHECK(hipFuncGetAttributes(&fa, shape->kernel));
    if (shape->tile_bytes + fa.sharedSizeBytes > prop.sharedMemPerBlockOptin) {
      fprintf(stderr, "LU tile %d B exceeds the device opt-in limit %zu\n", shape->tile_bytes,
              prop.sharedMemPerBlockOptin);
      return EXIT_FAILURE;
    }
  }
  HIP_CHECK(hipFuncSetAttribute(shape->kernel, hipFuncAttributeMaxDynamicSharedMemorySize,
                                shape->tile_bytes));
  if (zones <= 0) {
    int per_cu = 0;
    HIP_CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, shape->kernel, shape->block,
                                                           shape->tile_bytes));
    zones = prop.multiProcessorCount * std::max(per_cu, 1);
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

  printf("device: %s  arch: %s  compute units: %d  wavefront: %d\n", prop.name, prop.gcnArchName,
         prop.multiProcessorCount, prop.warpSize);
  printf("species: %d  reactions: %d  csr: %d  max_row: %d\n", NS, NR, (int)row_reac.size(),
         max_row);
  printf("block: %d  panel: %d  sub: %d  lds: %d\n", shape->block, shape->panel, shape->sub,
         shape->tile_bytes);
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

  std::vector<void*> allocs;
  // Copies n ints to the device.
  auto copy_i = [&](const int* src, size_t n) {
    int* d = nullptr;
    HIP_CHECK(hipMalloc(&d, sizeof(int) * n));
    HIP_CHECK(hipMemcpy(d, src, sizeof(int) * n, hipMemcpyHostToDevice));
    allocs.push_back(d);
    return d;
  };
  // Copies n doubles to the device.
  auto copy_d = [&](const double* src, size_t n) {
    double* d = nullptr;
    HIP_CHECK(hipMalloc(&d, sizeof(double) * n));
    HIP_CHECK(hipMemcpy(d, src, sizeof(double) * n, hipMemcpyHostToDevice));
    allocs.push_back(d);
    return d;
  };
  // Allocates n elements on the device.
  auto alloc = [&](auto*& p, size_t n) {
    HIP_CHECK(hipMalloc(&p, sizeof(*p) * n));
    allocs.push_back(p);
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

  void* args[] = {&dtab,  &d_row_off, &d_row_reac, &d_T,   &d_C,      &d_C0,
                  &d_w,   &d_rhs,     &d_J,        &d_perm, &d_kf,     &d_k0,
                  &d_logKc, &steps,   &dt,         &d_status, &d_niters, &d_part};
  // Enqueues one solver kernel.
  auto launch = [&]() {
    HIP_CHECK(hipLaunchKernel(shape->kernel, dim3(zones), dim3(shape->block), args,
                              shape->tile_bytes, 0));
  };

  struct Outcome {
    double seconds = 0;  // per timed launch
    int newton_bad = 0;
    int max_iters = 0;
    std::vector<double> states;
  };

  // Untimed launches, then timed ones, all continuing from state0. The timed
  // interval is host wall time through the launches and the wait for them.
  // states holds the result after every launch has run.
  auto exec = [&](int untimed, int timed) {
    Outcome o;
    HIP_CHECK(hipMemcpy(d_C, state0.data(), sizeof(double) * state0.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(d_status, 0, sizeof(int) * (size_t)zones));
    HIP_CHECK(hipMemset(d_niters, 0, sizeof(int) * (size_t)zones));
    for (int i = 0; i < untimed; ++i) launch();
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipGetLastError());
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < timed; ++i) launch();
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipGetLastError());
    const auto t1 = std::chrono::steady_clock::now();
    o.seconds = std::chrono::duration<double>(t1 - t0).count() / timed;
    std::vector<int> st(zones, 0), ni(zones, 0);
    HIP_CHECK(hipMemcpy(st.data(), d_status, sizeof(int) * (size_t)zones, hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(ni.data(), d_niters, sizeof(int) * (size_t)zones, hipMemcpyDeviceToHost));
    for (int z = 0; z < zones; ++z) {
      if (st[z] != 0) ++o.newton_bad;
      o.max_iters = std::max(o.max_iters, ni[z]);
    }
    o.states.resize(state0.size());
    HIP_CHECK(hipMemcpy(o.states.data(), d_C, sizeof(double) * state0.size(),
                        hipMemcpyDeviceToHost));
    if (o.newton_bad) printf("newton_fail full zones %d\n", o.newton_bad);
    return o;
  };

  // Frees the device buffers.
  auto cleanup = [&]() {
    for (void* p : allocs) HIP_CHECK(hipFree(p));
  };

  // One launch from state0, checked against the host reference before warmup
  // and timing. The host reference runs after that check and finishes before
  // the timed launches, so its OpenMP threads stay off the measurement.
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
  const bool check_ok =
      cpu_bad == 0 && !check.newton_bad && rel.finite && rel.max_rel <= kRelTol;
  if (!check_ok) {
    printf("relative-error %.3e FAIL\n", rel.max_rel);
    printf("stiff-ode: FAIL\n");
    cleanup();
    return EXIT_FAILURE;
  }

  Outcome timed = exec(warmup, iters);
  const bool pass = !timed.newton_bad;
  // RESULT block-size <threads> <seconds> s relative-error <max-rel> <PASS|FAIL>
  //   block-size: threads per block
  //   seconds: mean host wall time of one timed launch (STEPS backward-Euler steps)
  //   relative-error: largest relative error of the check launch against the host reference
  //   PASS: Newton succeeded on the timed launches; the check was already within 1e-12
  printf("RESULT block-size %d %.6f s relative-error %.3e %s\n", shape->block, timed.seconds,
         rel.max_rel, pass ? "PASS" : "FAIL");
  printf("speedup_vs_cpu: %.3fx\n", cpu_seconds / timed.seconds);
  printf("stiff-ode: %s\n", pass ? "PASS" : "FAIL");
  cleanup();
  return pass ? 0 : EXIT_FAILURE;
}

// Parses the arguments, picks a device, and runs the benchmark.
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

  int ndev = 0;
  HIP_CHECK(hipGetDeviceCount(&ndev));
  int dev = -1;
  int fallback = -1;
  for (int i = 0; i < ndev; ++i) {
    hipDeviceProp_t p;
    HIP_CHECK(hipGetDeviceProperties(&p, i));
    if (p.warpSize != 32 && p.warpSize != 64) continue;
    if (fallback < 0) fallback = i;
    if (dev < 0 && p.warpSize == 64 && p.multiProcessorCount >= 16) dev = i;
  }
  if (dev < 0) dev = fallback;
  if (dev < 0) {
    fprintf(stderr, "no device with a 32- or 64-wide wavefront\n");
    return EXIT_FAILURE;
  }
  HIP_CHECK(hipSetDevice(dev));
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, dev));
  return run(zones, iters, steps, warmup, dt_user, prop);
}
