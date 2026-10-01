#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include "../megablocks-cuda/reference.h"

using bf16 = sycl::ext::oneapi::bfloat16;
namespace mkl = oneapi::mkl;

// Per-group oneMKL parameters after megablocks' row-major operand swap:
// row-major C = op(a)*op(b) is computed column-major with the A/B operands
// swapped. The loop's first operand is megablocks' b, second is its a.
struct Groups {
  int E;
  std::vector<mkl::transpose> opA, opB;
  std::vector<int> M, N, K, lda, ldb, ldc;
  std::vector<size_t> offA, offB, offC;  // element offsets into b, a, c buffers
};

static const char* mode_name(GmmMode m) {
  switch (m) {
    case GMM_FWD:    return "fwd  (trans_a=0, trans_b=0)";
    case GMM_FWD_TB: return "fwd  (trans_a=0, trans_b=1)";
    default:         return "dw   (trans_a=1, variable K)";
  }
}

static Groups buildGroups(GmmMode mode, const std::vector<int>& batch_sizes,
                          int hidden_in, int hidden_out)
{
  const int E = (int)batch_sizes.size();
  const int N = hidden_out;
  Groups g;
  g.E = E;
  g.opA.resize(E); g.opB.resize(E);
  g.M.resize(E); g.N.resize(E); g.K.resize(E);
  g.lda.resize(E); g.ldb.resize(E); g.ldc.resize(E);
  g.offA.resize(E); g.offB.resize(E); g.offC.resize(E);

  size_t offA = 0, offB = 0, offC = 0;
  for (int e = 0; e < E; e++) {
    if (mode == GMM_FWD || mode == GMM_FWD_TB) {
      const int m_e = batch_sizes[e];
      const int K = hidden_in;
      const bool tb = (mode == GMM_FWD_TB);
      g.opA[e] = tb ? mkl::transpose::trans : mkl::transpose::nontrans;  // op on b
      g.opB[e] = mkl::transpose::nontrans;                                // op on a
      g.M[e]   = N;
      g.N[e]   = m_e;
      g.K[e]   = K;
      g.lda[e] = tb ? K : N;
      g.ldb[e] = K;
      g.ldc[e] = N;
      g.offB[e] = offB;  // b block  (Aptr operand)
      g.offA[e] = offA;  // a block  (Bptr operand)
      g.offC[e] = offC;
      offA += (size_t)m_e * K;
      offB += (size_t)K * N;
      offC += (size_t)m_e * N;
    } else { // GMM_DW: variable K
      const int k_e = batch_sizes[e];
      const int M = hidden_in;
      g.opA[e] = mkl::transpose::nontrans;  // op on b
      g.opB[e] = mkl::transpose::trans;     // op on a
      g.M[e]   = N;
      g.N[e]   = M;
      g.K[e]   = k_e;
      g.lda[e] = N;
      g.ldb[e] = M;
      g.ldc[e] = N;
      g.offB[e] = offB;
      g.offA[e] = offA;
      g.offC[e] = offC;
      offA += (size_t)k_e * M;
      offB += (size_t)k_e * N;
      offC += (size_t)M * N;
    }
  }

  // Variable-K (dw) only: megablocks sorts the grouped problems by descending K
  // (std::stable_sort on problem.k()) before launching. Replicate that ordering
  // here so the per-expert loop sees experts in the same order. Each group
  // carries its own buffer offsets/leading dimensions, so reordering the whole
  // tuple keeps the outputs landing in their original buffer blocks.
  if (mode == GMM_DW) {
    std::vector<int> idx(E);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(),
                     [&](int i, int j) { return g.K[i] > g.K[j]; });
    Groups s = g;
    for (int e = 0; e < E; e++) {
      s.opA[e] = g.opA[idx[e]]; s.opB[e] = g.opB[idx[e]];
      s.M[e]   = g.M[idx[e]];   s.N[e]   = g.N[idx[e]];   s.K[e]   = g.K[idx[e]];
      s.lda[e] = g.lda[idx[e]]; s.ldb[e] = g.ldb[idx[e]]; s.ldc[e] = g.ldc[idx[e]];
      s.offA[e] = g.offA[idx[e]]; s.offB[e] = g.offB[idx[e]]; s.offC[e] = g.offC[idx[e]];
    }
    g = std::move(s);
  }
  return g;
}

static void run_mode(sycl::queue& q, GmmMode mode,
                     const std::vector<int>& batch_sizes,
                     int hidden_in, int hidden_out, int repeat)
{
  const int E = (int)batch_sizes.size();

  size_t nA = 0, nB = 0, nC = 0;
  for (int e = 0; e < E; e++) {
    if (mode == GMM_FWD || mode == GMM_FWD_TB) {
      nA += (size_t)batch_sizes[e] * hidden_in;
      nB += (size_t)hidden_in * hidden_out;
      nC += (size_t)batch_sizes[e] * hidden_out;
    } else {
      nA += (size_t)batch_sizes[e] * hidden_in;
      nB += (size_t)batch_sizes[e] * hidden_out;
      nC += (size_t)hidden_in * hidden_out;
    }
  }

  printf("---- mode: %s ----\n", mode_name(mode));

  std::vector<float> hAf(nA), hBf(nB), hRef(nC, 0.f);
  std::vector<bf16>  hA(nA), hB(nB);
  srand48(123);
  for (size_t i = 0; i < nA; i++) { hAf[i] = (float)drand48() - 0.5f; hA[i] = bf16(hAf[i]); }
  for (size_t i = 0; i < nB; i++) { hBf[i] = (float)drand48() - 0.5f; hB[i] = bf16(hBf[i]); }

  bf16* dA = sycl::malloc_device<bf16>(nA, q);
  bf16* dB = sycl::malloc_device<bf16>(nB, q);
  bf16* dC = sycl::malloc_device<bf16>(nC, q);
  q.memcpy(dA, hA.data(), nA * sizeof(bf16));
  q.memcpy(dB, hB.data(), nB * sizeof(bf16));
  q.wait();

  Groups g = buildGroups(mode, batch_sizes, hidden_in, hidden_out);

  const float alpha = 1.f, beta = 0.f;

  // Skip the O(sum m*n*k) host reference for large problems.
  double total_flop = gmm_gflops(mode, batch_sizes, hidden_in, hidden_out, 1.0) * 1e3;
  const bool do_verify = total_flop < 2e10;
  const double tol = 1.0;
  std::vector<float> hCf(nC);
  if (do_verify)
    gmm_ref(mode, batch_sizes, hidden_in, hidden_out, hAf, hBf, hRef);
  else
    printf("  (host verification skipped: problem too large)\n");

  // Device output is bf16; read it back and widen to float for comparison.
  auto readback_convert = [&]() {
    std::vector<bf16> hC(nC);
    q.memcpy(hC.data(), dC, nC * sizeof(bf16)).wait();
    for (size_t i = 0; i < nC; i++) hCf[i] = (float)hC[i];
  };

  // ---------------------------------------------------------------------------
  // Backend 1: per-expert oneMKL gemm loop.
  // ---------------------------------------------------------------------------
  auto launch_loop = [&]() {
    for (int e = 0; e < E; e++) {
      // Empty expert (0 tokens => m/n or variable-K == 0): the GEMM is a no-op
      // and its output is already zeroed by the memset below.
      // Skip the degenerate zero-dim gemm.
      if (g.M[e] == 0 || g.N[e] == 0 || g.K[e] == 0) continue;
      mkl::blas::column_major::gemm(
          q, g.opA[e], g.opB[e],
          g.M[e], g.N[e], g.K[e],
          alpha,
          dB + g.offB[e], g.lda[e],   // first operand = megablocks' b
          dA + g.offA[e], g.ldb[e],   // second operand = megablocks' a
          beta,
          dC + g.offC[e], g.ldc[e]);
    }
  };

  q.memset(dC, 0, nC * sizeof(bf16)).wait();
  for (int r = 0; r < 30; r++) launch_loop();
  q.wait();
  printf("  [oneMKL per-expert loop     ]\n");
  if (do_verify) { readback_convert(); verify(hCf, hRef, tol); }

  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) launch_loop();
  q.wait();
  auto t1 = std::chrono::steady_clock::now();
  double loop_us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-3 / repeat;
  double loop_tf = gmm_gflops(mode, batch_sizes, hidden_in, hidden_out, loop_us) / 1000.0;
  printf("  time: %.3f us | %.2f TFLOP/s\n", loop_us, loop_tf);

  // ---------------------------------------------------------------------------
  // Backend 2: single-launch oneMKL group gemm_batch (mirrors the CUDA path
  // built on cublasGemmGroupedBatchedEx). Each expert is its own group of size
  // 1 since the per-expert shapes differ. Empty experts are dropped from the
  // group list; their outputs stay zeroed by the memset above.
  // ---------------------------------------------------------------------------
  std::vector<mkl::transpose> fopA, fopB;
  std::vector<std::int64_t> fM, fN, fK, flda, fldb, fldc, fgrp;
  std::vector<size_t> foffA, foffB, foffC;
  for (int e = 0; e < E; e++) {
    if (g.M[e] == 0 || g.N[e] == 0 || g.K[e] == 0) continue;
    fopA.push_back(g.opA[e]);   fopB.push_back(g.opB[e]);
    fM.push_back(g.M[e]);       fN.push_back(g.N[e]);       fK.push_back(g.K[e]);
    flda.push_back(g.lda[e]);   fldb.push_back(g.ldb[e]);   fldc.push_back(g.ldc[e]);
    foffB.push_back(g.offB[e]); foffA.push_back(g.offA[e]); foffC.push_back(g.offC[e]);
    fgrp.push_back(1);
  }
  const std::int64_t Eg = (std::int64_t)fM.size();

  // Pointer arrays for the group API live in shared memory (host-populated,
  // device-readable). Element order matches the megablocks operand swap: the
  // grouped "A" operand is b, the "B" operand is a.
  const bf16** Aarr = sycl::malloc_shared<const bf16*>(Eg, q);
  const bf16** Barr = sycl::malloc_shared<const bf16*>(Eg, q);
  bf16**       Carr = sycl::malloc_shared<bf16*>(Eg, q);
  for (std::int64_t e = 0; e < Eg; e++) {
    Aarr[e] = dB + foffB[e];
    Barr[e] = dA + foffA[e];
    Carr[e] = dC + foffC[e];
  }
  std::vector<float> alpha_arr(Eg, alpha), beta_arr(Eg, beta);

  auto launch_grouped = [&]() {
    mkl::blas::column_major::gemm_batch(
        q, fopA.data(), fopB.data(),
        fM.data(), fN.data(), fK.data(),
        alpha_arr.data(),
        Aarr, flda.data(),
        Barr, fldb.data(),
        beta_arr.data(),
        Carr, fldc.data(),
        Eg, fgrp.data());
  };

  q.memset(dC, 0, nC * sizeof(bf16)).wait();
  for (int r = 0; r < 30; r++) launch_grouped();
  q.wait();
  printf("  [oneMKL group gemm_batch    ]\n");
  if (do_verify) { readback_convert(); verify(hCf, hRef, tol); }

  auto s0 = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) launch_grouped();
  q.wait();
  auto s1 = std::chrono::steady_clock::now();
  double grp_us = std::chrono::duration_cast<std::chrono::nanoseconds>(s1 - s0).count() * 1e-3 / repeat;
  double grp_tf = gmm_gflops(mode, batch_sizes, hidden_in, hidden_out, grp_us) / 1000.0;
  printf("  time: %.3f us | %.2f TFLOP/s | grouped/loop speedup: %.2fx\n",
         grp_us, grp_tf, loop_us / grp_us);

  sycl::free(Aarr, q);
  sycl::free(Barr, q);
  sycl::free(Carr, q);
  sycl::free(dA, q);
  sycl::free(dB, q);
  sycl::free(dC, q);
}

int main(int argc, char* argv[])
{
  const int repeat      = (argc > 1) ? atoi(argv[1]) : 100;
  const int num_experts = (argc > 2) ? atoi(argv[2]) : 64;
  const int hidden_in   = (argc > 3) ? atoi(argv[3]) : 2048;
  const int hidden_out  = (argc > 4) ? atoi(argv[4]) : 2048;
  const int avg_tokens  = (argc > 5) ? atoi(argv[5]) : 16;

  const long num_tokens = (long)num_experts * avg_tokens;
  srand48(123);
  std::vector<double> cdf(num_experts);
  double wsum = 0;
  for (int e = 0; e < num_experts; e++) { wsum += 0.2 + drand48(); cdf[e] = wsum; }
  std::vector<int> batch_sizes(num_experts, 0);
  for (long t = 0; t < num_tokens; t++) {
    double r = drand48() * wsum;
    int e = 0;
    while (e < num_experts - 1 && r > cdf[e]) e++;
    batch_sizes[e]++;
  }
  // MoE routing leaves some experts with zero tokens (batch_sizes[e] == 0).
  int empty_experts = 0;
  for (int e = 0; e < num_experts; e++) if (batch_sizes[e] == 0) empty_experts++;

  printf("megablocks grouped GEMM: %d experts, hidden_in=%d, hidden_out=%d, "
         "avg_tokens/expert=%d, repeat=%d (bfloat16); %d empty expert(s)\n",
         num_experts, hidden_in, hidden_out, avg_tokens, repeat, empty_experts);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  run_mode(q, GMM_FWD,    batch_sizes, hidden_in, hidden_out, repeat);
  run_mode(q, GMM_FWD_TB, batch_sizes, hidden_in, hidden_out, repeat);
  run_mode(q, GMM_DW,     batch_sizes, hidden_in, hidden_out, repeat);

  return 0;
}
