// Batched 128x128 SPD inverse with two epilogue products.
//
//   T = diag(alpha) * (I + A)^{-1}
//   J = B * (I + A)^{-1}
//
// Workload and acceptance follow the Video Delta rule kernel in
// https://github.com/NVlabs/kda/issues/6 . A is symmetric PSD, everything is
// fp32, and TF32 or a narrower type is rejected because it destroys the
// conditioning of I+A. d is 128. frames and heads only set how many matrices
// the grid launches; the kernel is not templated on them.
//
// Three results are checked, each against an fp64 general inverse of I+A:
//   reference.h   the four torch calls, on the CPU, in fp32
//   fused.cu      one block per matrix, one launch
//   libraries.cu  hipsolver potrf, hipblas trsm, and the two products
// Pass means finite and a Frobenius relative error of at most 1e-6 on T and
// on J. Timing is std::chrono around the launches plus one device sync after
// the repeat loop, not HIP events.

#include <hip/hip_runtime.h>

#include "fused.h"
#include "libraries.h"
#include "reference.h"

#include "host_common.hpp"
#include "local_layout.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_CHECK(call)                                                        \
  do {                                                                         \
    const hipError_t error_ = (call);                                          \
    if (error_ != hipSuccess) {                                                \
      fprintf(stderr, "HIP error at %s:%d: %s failed: %s\n",                   \
              __FILE__, __LINE__, #call, hipGetErrorString(error_));           \
      return EXIT_FAILURE;                                                     \
    }                                                                          \
  } while (0)

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

  hipDeviceProp_t prop{};
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  // The device query that fused_layout runs also sets fused_wave.
  const char* layout = fused_layout();
  const size_t smem_max = prop.sharedMemPerBlockOptin > prop.sharedMemPerBlock
                              ? prop.sharedMemPerBlockOptin
                              : prop.sharedMemPerBlock;
  printf("GPU: %s, %d CUs, shared memory %zu bytes, wave %d, %s\n",
         prop.name, prop.multiProcessorCount, smem_max, fused_wave(), layout);
  if (smem_max < kVdiTightBytes) {
    fprintf(stderr, "need %zu bytes of dynamic shared memory\n", kVdiTightBytes);
    return EXIT_FAILURE;
  }
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

  float *dA = nullptr, *dB = nullptr, *dAlpha = nullptr, *dT = nullptr, *dJ = nullptr;
  HIP_CHECK(hipMalloc(&dA, ntot * sizeof(float)));
  HIP_CHECK(hipMalloc(&dB, ntot * sizeof(float)));
  HIP_CHECK(hipMalloc(&dAlpha, alpha.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&dT, ntot * sizeof(float)));
  HIP_CHECK(hipMalloc(&dJ, ntot * sizeof(float)));
  HIP_CHECK(hipMemcpy(dA, A.data(), ntot * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB, B.data(), ntot * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dAlpha, alpha.data(), alpha.size() * sizeof(float),
                      hipMemcpyHostToDevice));

  if (fused_init(nmat) != 0) return EXIT_FAILURE;

  auto launch_fused = [&]() { return fused_launch(dA, dB, dAlpha, dT, dJ, nmat); };

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

  if (libraries_init(nmat) != 0) return EXIT_FAILURE;

  hipError_t errc = hipSuccess;
  auto fail = [&](const char* what) {
    fprintf(stderr, "HIP error: %s failed: %s\n", what, hipGetErrorString(errc));
    libraries_shutdown();
    fused_shutdown();
    return EXIT_FAILURE;
  };
  if (launch_fused() != 0) {
    libraries_shutdown();
    fused_shutdown();
    return EXIT_FAILURE;
  }
  errc = hipDeviceSynchronize();
  if (errc != hipSuccess) return fail("hipDeviceSynchronize");
  errc = hipGetLastError();
  if (errc != hipSuccess) return fail("fused kernel");
  errc = hipMemcpy(Tgpu.data(), dT, ntot * sizeof(float), hipMemcpyDeviceToHost);
  if (errc != hipSuccess) return fail("hipMemcpy T");
  errc = hipMemcpy(Jgpu.data(), dJ, ntot * sizeof(float), hipMemcpyDeviceToHost);
  if (errc != hipSuccess) return fail("hipMemcpy J");

  const int pivot_bad = fused_pivot_failed();
  if (pivot_bad) fprintf(stderr, "hip: fused kernel         non-positive pivot\n");
  const ErrorStats err = compare(Tgpu.data(), Jgpu.data(), T64.data(), J64.data(),
                                 alpha.data(), nmat);
  const bool hip_ok = !pivot_bad && err.finite && err.rel_t <= kRelTol && err.rel_j <= kRelTol &&
                      err.asym <= 1e-4;
  printf("hip: fused kernel         rel err vs fp64  T %.3e  J %.3e  max|d| %.3e/%.3e  "
         "asym %.3e  %s\n",
         err.rel_t, err.rel_j, err.max_t, err.max_j, err.asym,
         hip_ok ? "OK" : "FAIL");
  if (err.asym > 1e-4) {
    printf("  worst T/alpha pair: alpha %.3e %.3e  T %.3e %.3e  inv %.3e %.3e\n",
           err.worst_ar, err.worst_ac, err.worst_trc, err.worst_tcr,
           err.worst_trc / err.worst_ar, err.worst_tcr / err.worst_ac);
  }

  bool lib_ok = false;
  if (libraries_launch(dA, dB, dAlpha, dT, dJ, nmat, 1) != 0) {
    printf("hip: libraries             launch failed\n");
  } else {
    errc = hipMemcpy(Tlib.data(), dT, ntot * sizeof(float), hipMemcpyDeviceToHost);
    if (errc != hipSuccess) return fail("hipMemcpy T lib");
    errc = hipMemcpy(Jlib.data(), dJ, ntot * sizeof(float), hipMemcpyDeviceToHost);
    if (errc != hipSuccess) return fail("hipMemcpy J lib");
    const ErrorStats lerr = compare(Tlib.data(), Jlib.data(), T64.data(), J64.data(),
                                    alpha.data(), nmat);
    lib_ok = lerr.finite && lerr.rel_t <= kRelTol && lerr.rel_j <= kRelTol &&
             lerr.asym <= 1e-4;
    printf("hip: libraries             rel err vs fp64  T %.3e  J %.3e  max|d| %.3e/%.3e  "
           "asym %.3e  %s\n",
           lerr.rel_t, lerr.rel_j, lerr.max_t, lerr.max_j, lerr.asym,
           lib_ok ? "OK" : "FAIL");
  }

  // Warmup is synchronized before the clock starts. The clock covers the
  // launches only; one sync after the loop completes the last call. Library
  // launches pass check_info=0, so they do not add their own sync.
  auto time_ms = [&](auto&& launch) -> double {
    for (int i = 0; i < kWarmup; ++i)
      if (launch() != 0) return -1.0;
    errc = hipDeviceSynchronize();
    if (errc != hipSuccess) return -1.0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < repeat; ++i)
      if (launch() != 0) return -1.0;
    errc = hipDeviceSynchronize();
    if (errc != hipSuccess) return -1.0;
    return std::chrono::duration<double, std::milli>(
               std::chrono::steady_clock::now() - t0)
               .count() /
           repeat;
  };

  if (hip_ok) {
    const double ms = time_ms(launch_fused);
    if (ms < 0.0) return fail("fused timing");
    printf("hip fused: %.4f ms/call, 1 launch, %.1f ms/forward (x50)\n",
           ms, ms * 50.0);
  }
  if (lib_ok) {
    const double ms = time_ms([&]() {
      return libraries_launch(dA, dB, dAlpha, dT, dJ, nmat, 0);
    });
    if (ms < 0.0) return fail("library timing");
    printf("hip libraries: %.4f ms/call, potrf + trsm + gemm, %.1f ms/forward (x50)\n",
           ms, ms * 50.0);
  }

  // The line "PASS" is what `make run` looks for. All three paths must clear
  // 1e-6, and both GPU paths must also clear the symmetry check.
  const bool all_ok = today_pass && hip_ok && lib_ok;

  fused_shutdown();
  libraries_shutdown();
  (void)hipFree(dA);
  (void)hipFree(dB);
  (void)hipFree(dAlpha);
  (void)hipFree(dT);
  (void)hipFree(dJ);

  printf("%s\n", all_ok ? "PASS" : "FAIL");
  return all_ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
