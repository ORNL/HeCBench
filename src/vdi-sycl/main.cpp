// Batched 128x128 SPD inverse with two epilogue products.
//
//   T = diag(alpha) * (I + A)^{-1}
//   J = B * (I + A)^{-1}
//
// Workload and acceptance follow the Video Delta rule kernel in
// https://github.com/NVlabs/kda/issues/6 . A is symmetric PSD, everything is
// fp32, and TF32 or a narrower type is rejected because it destroys the
// conditioning of I+A. d is 128. frames and heads only set how many matrices
// the nd_range launches; the kernel is not templated on them.
//
// Three results are checked, each against an fp64 general inverse of I+A:
//   reference.h    the four torch calls, on the CPU, in fp32
//   fused.cpp      one work-group per matrix, one launch
//   libraries.cpp  oneMKL potrf, trsm, and the two products
// Pass means finite and a Frobenius relative error of at most 1e-6 on T and
// on J. Timing is std::chrono around the launches plus one queue wait after
// the repeat loop, not SYCL event profiling.
//
// oneMKL's SYCL device code is Intel SPIR-V. On a device it has no image for,
// libraries_init returns 2, the library line reads "skipped", and PASS is
// decided by the CPU reference and the fused kernel.

#include <sycl/sycl.hpp>

#include "fused.h"
#include "libraries.h"
#include "reference.h"

#include "host_common.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  int frames = kFrames;
  int heads = kHeads;
  int tokens = kDefaultTokens;
  int dim = kDim;
  int repeat = 50;
  float beta_scale = 1.0f;
  int positional = 0;
  for (int i = 1; i < argc; ++i) {
    const auto take = [&](const char* flag) -> const char* {
      if (i + 1 >= argc) {
        fprintf(stderr, "missing value for %s\n", flag);
        return nullptr;
      }
      return argv[++i];
    };
    const char* v = nullptr;
    if (std::strcmp(argv[i], "--frames") == 0) {
      v = take("--frames");
      if (!v) return EXIT_FAILURE;
      frames = std::atoi(v);
    } else if (std::strcmp(argv[i], "--heads") == 0) {
      v = take("--heads");
      if (!v) return EXIT_FAILURE;
      heads = std::atoi(v);
    } else if (std::strcmp(argv[i], "--tokens") == 0) {
      v = take("--tokens");
      if (!v) return EXIT_FAILURE;
      tokens = std::atoi(v);
    } else if (std::strcmp(argv[i], "--d") == 0) {
      v = take("--d");
      if (!v) return EXIT_FAILURE;
      dim = std::atoi(v);
    } else if (std::strcmp(argv[i], "--beta-scale") == 0) {
      v = take("--beta-scale");
      if (!v) return EXIT_FAILURE;
      beta_scale = std::atof(v);
    } else if (std::strcmp(argv[i], "--repeat") == 0) {
      v = take("--repeat");
      if (!v) return EXIT_FAILURE;
      repeat = std::atoi(v);
    } else if (argv[i][0] != '-') {
      if (positional == 0) frames = std::atoi(argv[i]);
      else if (positional == 1) heads = std::atoi(argv[i]);
      else {
        fprintf(stderr, "unexpected argument %s\n", argv[i]);
        return EXIT_FAILURE;
      }
      ++positional;
    } else {
      fprintf(stderr,
              "usage: %s [frames heads] [--frames 101] [--heads 7] [--tokens 1008] "
              "[--d 128] [--beta-scale 1] [--repeat 50]\n",
              argv[0]);
      return EXIT_FAILURE;
    }
  }
  if (frames < 1 || heads < 1 || tokens < 1 || repeat < 1 || beta_scale <= 0.0f) {
    fprintf(stderr, "frames, heads, tokens, repeat, and beta-scale must be positive\n");
    return EXIT_FAILURE;
  }
  if (dim != kDim) {
    fprintf(stderr, "d=%d is not supported; this kernel is built for d=%d\n", dim, kDim);
    return EXIT_FAILURE;
  }
  const int nmat = frames * heads;

  if (!self_check_reference()) return EXIT_FAILURE;

  try {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
    const sycl::device dev = q.get_device();
    const std::string dev_name = dev.get_info<sycl::info::device::name>();

    // fused_prepare rejects a device whose sub-group or local memory does
    // not fit this binary's kernel, the role of CUDA's opt-in check.
    if (fused_prepare(dev) != 0) return EXIT_FAILURE;
    printf("GPU: %s, %u compute units, local memory %zu bytes, wave %d, %s\n",
           dev_name.c_str(),
           static_cast<unsigned>(dev.get_info<sycl::info::device::max_compute_units>()),
           static_cast<size_t>(dev.get_info<sycl::info::device::local_mem_size>()),
           fused_wave(), fused_layout());
    printf("workload: N=%d (frames=%d heads=%d) d=%d tokens=%d beta-scale=%g repeat=%d warmup=%d\n",
           nmat, frames, heads, kDim, tokens, beta_scale, repeat, kWarmup);
    const size_t nn = static_cast<size_t>(kDim) * kDim;
    const size_t ntot = static_cast<size_t>(nmat) * nn;
    std::vector<float> A(ntot), B(ntot), alpha(static_cast<size_t>(nmat) * kDim);
    std::vector<float> Ttoday(ntot), Jtoday(ntot), Tgpu(ntot), Jgpu(ntot);
    std::vector<float> Tlib(ntot), Jlib(ntot);
    std::vector<double> Ad(ntot), Bd(ntot), alphad(alpha.size());
    std::vector<double> T64(ntot), J64(ntot);

    printf("generating inputs...\n");
    fflush(stdout);
    const auto gen0 = std::chrono::steady_clock::now();
    generate_inputs(nmat, heads, tokens, beta_scale, A, B, alpha);
    const double gen_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - gen0).count();
    printf("input generation: %.2f s\n", gen_s);
    for (size_t i = 0; i < ntot; ++i) {
      Ad[i] = A[i];
      Bd[i] = B[i];
    }
    for (size_t i = 0; i < alpha.size(); ++i) alphad[i] = alpha[i];

    float* dA = sycl::malloc_device<float>(ntot, q);
    float* dB = sycl::malloc_device<float>(ntot, q);
    float* dAlpha = sycl::malloc_device<float>(alpha.size(), q);
    float* dT = sycl::malloc_device<float>(ntot, q);
    float* dJ = sycl::malloc_device<float>(ntot, q);
    auto free_device = [&]() {
      if (dA) sycl::free(dA, q);
      if (dB) sycl::free(dB, q);
      if (dAlpha) sycl::free(dAlpha, q);
      if (dT) sycl::free(dT, q);
      if (dJ) sycl::free(dJ, q);
    };
    if (!dA || !dB || !dAlpha || !dT || !dJ) {
      fprintf(stderr, "SYCL error: malloc_device failed\n");
      free_device();
      return EXIT_FAILURE;
    }
    q.memcpy(dA, A.data(), ntot * sizeof(float));
    q.memcpy(dB, B.data(), ntot * sizeof(float));
    q.memcpy(dAlpha, alpha.data(), alpha.size() * sizeof(float));
    q.wait_and_throw();

    if (fused_init(q, nmat) != 0) {
      free_device();
      return EXIT_FAILURE;
    }

    auto launch_fused = [&]() { return fused_launch(q, dA, dB, dAlpha, dT, dJ, nmat); };

    printf("fp64 inv reference...\n");
    fflush(stdout);
    const auto ref0 = std::chrono::steady_clock::now();
    const bool fp64_ok = fp64_batch(Ad.data(), Bd.data(), alphad.data(), T64.data(),
                                    J64.data(), nmat, kDim);
    const double ref_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - ref0).count();
    printf("fp64 inv reference: %.2f s for %d matrices (%.2f ms/matrix)\n",
           ref_s, nmat, ref_s * 1e3 / nmat);
    if (!fp64_ok) {
      fprintf(stderr, "fp64 inv failed\n");
      fused_shutdown(q);
      free_device();
      return EXIT_FAILURE;
    }

    printf("four-call CPU reference (cholesky + trsm + gemm)...\n");
    fflush(stdout);
    const auto today0 = std::chrono::steady_clock::now();
    today_batch(A.data(), B.data(), alpha.data(), Ttoday.data(), Jtoday.data(),
                nmat, kDim);
    const double today_s = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - today0).count();
    printf("four-call CPU reference: %.2f s for %d matrices (%.2f ms/matrix)\n",
           today_s, nmat, today_s * 1e3 / nmat);
    const ErrorStats today_err = compare(Ttoday.data(), Jtoday.data(), T64.data(),
                                         J64.data(), alpha.data(), nmat);
    const bool today_pass = today_err.finite && today_err.rel_t <= kRelTol &&
                            today_err.rel_j <= kRelTol;
    printf("today: cholesky+trsm+gemm  rel err vs fp64  T %.3e  J %.3e  max|d| %.3e/%.3e  %s\n",
           today_err.rel_t, today_err.rel_j, today_err.max_t, today_err.max_j,
           today_pass ? "OK" : "FAIL");

    const int lib_init = libraries_init(q, nmat);
    if (lib_init == 1) {
      fused_shutdown(q);
      free_device();
      return EXIT_FAILURE;
    }
    const bool lib_skipped = lib_init == 2;

    auto fail = [&](const char* what) {
      fprintf(stderr, "SYCL error: %s failed\n", what);
      libraries_shutdown();
      fused_shutdown(q);
      free_device();
      return EXIT_FAILURE;
    };
    auto sync = [&]() {
      try {
        q.wait_and_throw();
      } catch (const sycl::exception& ex) {
        fprintf(stderr, "SYCL error: %s\n", ex.what());
        return false;
      }
      return true;
    };

    if (launch_fused() != 0) return fail("fused launch");
    if (!sync()) return fail("fused kernel");
    q.memcpy(Tgpu.data(), dT, ntot * sizeof(float));
    q.memcpy(Jgpu.data(), dJ, ntot * sizeof(float));
    if (!sync()) return fail("memcpy T/J");

    const int pivot_bad = fused_pivot_failed(q);
    if (pivot_bad) fprintf(stderr, "sycl: fused kernel         non-positive pivot\n");
    const ErrorStats err = compare(Tgpu.data(), Jgpu.data(), T64.data(), J64.data(),
                                   alpha.data(), nmat);
    const bool sycl_ok = !pivot_bad && err.finite && err.rel_t <= kRelTol &&
                         err.rel_j <= kRelTol && err.asym <= 1e-4;
    printf("sycl: fused kernel         rel err vs fp64  T %.3e  J %.3e  max|d| %.3e/%.3e  "
           "asym %.3e  %s\n",
           err.rel_t, err.rel_j, err.max_t, err.max_j, err.asym,
           sycl_ok ? "OK" : "FAIL");
    if (err.asym > 1e-4) {
      printf("  worst T/alpha pair: alpha %.3e %.3e  T %.3e %.3e  inv %.3e %.3e\n",
             err.worst_ar, err.worst_ac, err.worst_trc, err.worst_tcr,
             err.worst_trc / err.worst_ar, err.worst_tcr / err.worst_ac);
    }

    bool lib_ok = false;
    if (lib_skipped) {
      printf("sycl: libraries             skipped\n");
    } else if (libraries_launch(q, dA, dB, dAlpha, dT, dJ, nmat, 1) != 0 || !sync()) {
      printf("sycl: libraries             launch failed\n");
    } else {
      q.memcpy(Tlib.data(), dT, ntot * sizeof(float));
      q.memcpy(Jlib.data(), dJ, ntot * sizeof(float));
      if (!sync()) return fail("memcpy T/J lib");
      const ErrorStats lerr = compare(Tlib.data(), Jlib.data(), T64.data(), J64.data(),
                                      alpha.data(), nmat);
      lib_ok = lerr.finite && lerr.rel_t <= kRelTol && lerr.rel_j <= kRelTol &&
               lerr.asym <= 1e-4;
      printf("sycl: libraries             rel err vs fp64  T %.3e  J %.3e  max|d| %.3e/%.3e  "
             "asym %.3e  %s\n",
             lerr.rel_t, lerr.rel_j, lerr.max_t, lerr.max_j, lerr.asym,
             lib_ok ? "OK" : "FAIL");
    }

    // Warmup is synchronized before the clock starts. The clock covers the
    // launches only; one wait after the loop completes the last call. Library
    // launches pass check_info=0, so they do not add their own wait.
    auto time_ms = [&](auto&& launch) -> double {
      for (int i = 0; i < kWarmup; ++i)
        if (launch() != 0) return -1.0;
      if (!sync()) return -1.0;
      const auto t0 = std::chrono::steady_clock::now();
      for (int i = 0; i < repeat; ++i)
        if (launch() != 0) return -1.0;
      if (!sync()) return -1.0;
      return std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - t0)
                 .count() /
             repeat;
    };

    if (sycl_ok) {
      const double ms = time_ms(launch_fused);
      if (ms < 0.0) return fail("fused timing");
      printf("sycl fused: %.4f ms/call, 1 launch, %.1f ms/forward (x50)\n",
             ms, ms * 50.0);
    }
    if (lib_ok) {
      const double ms = time_ms([&]() {
        return libraries_launch(q, dA, dB, dAlpha, dT, dJ, nmat, 0);
      });
      if (ms < 0.0) return fail("library timing");
      printf("sycl libraries: %.4f ms/call, potrf + trsm + gemm, %.1f ms/forward (x50)\n",
             ms, ms * 50.0);
    }

    // The line "PASS" is what `make run` looks for. The CPU reference and the
    // fused kernel must clear 1e-6, the fused kernel must also clear the
    // symmetry check, and the library path must pass unless it was skipped.
    const bool all_ok = today_pass && sycl_ok && (lib_ok || lib_skipped);

    fused_shutdown(q);
    libraries_shutdown();
    free_device();

    printf("%s\n", all_ok ? "PASS" : "FAIL");
    return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const sycl::exception& ex) {
    fprintf(stderr, "SYCL error: %s\n", ex.what());
    return EXIT_FAILURE;
  }
}
