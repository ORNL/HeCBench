#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <hip/hip_runtime.h>
#include <hip/hip_bfloat16.h>
#define HIPBLAS_V2
#include <hipblas/hipblas.h>
#include <hipblaslt/hipblaslt.h>
#include <hipblaslt/hipblaslt-ext.hpp>
#include "../megablocks-cuda/reference.h"

#define CHECK_HIP(call)                                                      \
  do {                                                                       \
    hipError_t err = (call);                                                 \
    if (err != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error %s at %s:%d\n",                             \
              hipGetErrorString(err), __FILE__, __LINE__);                   \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

#define CHECK_HIPBLAS(call)                                                  \
  do {                                                                       \
    hipblasStatus_t st = (call);                                             \
    if (st != HIPBLAS_STATUS_SUCCESS) {                                      \
      fprintf(stderr, "hipBLAS error %d at %s:%d\n", st, __FILE__, __LINE__);\
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

using bf16 = hip_bfloat16;

// Per-group hipBLAS parameters after megablocks' row-major operand swap:
// row-major C = op(a)*op(b) is computed by hipblas as column-major with the A/B
// operands swapped. So the loop's "A" operand is megablocks' b, "B" is its a.
struct Groups {
  int E;
  std::vector<hipblasOperation_t> opA, opB;
  std::vector<int> M, N, K, lda, ldb, ldc;
  std::vector<const void*> Aptr;  // -> b blocks
  std::vector<const void*> Bptr;  // -> a blocks
  std::vector<void*> Cptr;        // -> c blocks
};

static const char* mode_name(GmmMode m) {
  switch (m) {
    case GMM_FWD:    return "fwd  (trans_a=0, trans_b=0)";
    case GMM_FWD_TB: return "fwd  (trans_a=0, trans_b=1)";
    default:         return "dw   (trans_a=1, variable K)";
  }
}

static Groups buildGroups(GmmMode mode, const std::vector<int>& batch_sizes,
                          int hidden_in, int hidden_out,
                          bf16* dA, bf16* dB, bf16* dC)
{
  const int E = (int)batch_sizes.size();
  const int N = hidden_out;
  Groups g;
  g.E = E;
  g.opA.resize(E); g.opB.resize(E);
  g.M.resize(E); g.N.resize(E); g.K.resize(E);
  g.lda.resize(E); g.ldb.resize(E); g.ldc.resize(E);
  g.Aptr.resize(E); g.Bptr.resize(E); g.Cptr.resize(E);

  size_t offA = 0, offB = 0, offC = 0;
  for (int e = 0; e < E; e++) {
    if (mode == GMM_FWD || mode == GMM_FWD_TB) {
      const int m_e = batch_sizes[e];
      const int K = hidden_in;
      const bool tb = (mode == GMM_FWD_TB);
      g.opA[e] = tb ? HIPBLAS_OP_T : HIPBLAS_OP_N;  // op on b
      g.opB[e] = HIPBLAS_OP_N;                       // op on a
      g.M[e]   = N;
      g.N[e]   = m_e;
      g.K[e]   = K;
      g.lda[e] = tb ? K : N;
      g.ldb[e] = K;
      g.ldc[e] = N;
      g.Aptr[e] = dB + offB;
      g.Bptr[e] = dA + offA;
      g.Cptr[e] = dC + offC;
      offA += (size_t)m_e * K;
      offB += (size_t)K * N;
      offC += (size_t)m_e * N;
    } else { // GMM_DW: variable K
      const int k_e = batch_sizes[e];
      const int M = hidden_in;
      g.opA[e] = HIPBLAS_OP_N;  // op on b
      g.opB[e] = HIPBLAS_OP_T;  // op on a
      g.M[e]   = N;
      g.N[e]   = M;
      g.K[e]   = k_e;
      g.lda[e] = N;
      g.ldb[e] = M;
      g.ldc[e] = N;
      g.Aptr[e] = dB + offB;
      g.Bptr[e] = dA + offA;
      g.Cptr[e] = dC + offC;
      offA += (size_t)k_e * M;
      offB += (size_t)k_e * N;
      offC += (size_t)M * N;
    }
  }

  // Variable-K (dw) only: megablocks sorts the grouped problems by descending K
  // (std::stable_sort on problem.k()) before launching. Replicate that ordering
  // here so the per-expert loop sees experts in the same order. Each group
  // carries its own pointers/leading dimensions, so reordering the whole tuple
  // keeps the outputs landing in their original device blocks.
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
      s.Aptr[e] = g.Aptr[idx[e]]; s.Bptr[e] = g.Bptr[idx[e]]; s.Cptr[e] = g.Cptr[idx[e]];
    }
    g = std::move(s);
  }
  return g;
}

static void run_mode(GmmMode mode, const std::vector<int>& batch_sizes,
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

  bf16 *dA, *dB, *dC;
  CHECK_HIP(hipMalloc(&dA, nA * sizeof(bf16)));
  CHECK_HIP(hipMalloc(&dB, nB * sizeof(bf16)));
  CHECK_HIP(hipMalloc(&dC, nC * sizeof(bf16)));
  CHECK_HIP(hipMemcpy(dA, hA.data(), nA * sizeof(bf16), hipMemcpyHostToDevice));
  CHECK_HIP(hipMemcpy(dB, hB.data(), nB * sizeof(bf16), hipMemcpyHostToDevice));

  Groups g = buildGroups(mode, batch_sizes, hidden_in, hidden_out, dA, dB, dC);

  hipblasHandle_t handle;
  CHECK_HIPBLAS(hipblasCreate(&handle));
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

  auto readback_convert = [&]() {
    std::vector<bf16> hC(nC);
    CHECK_HIP(hipMemcpy(hC.data(), dC, nC * sizeof(bf16), hipMemcpyDeviceToHost));
    for (size_t i = 0; i < nC; i++) hCf[i] = float(hC[i]);
  };

  // per-expert hipblasGemmEx loop.
  auto launch_loop = [&]() {
    for (int e = 0; e < E; e++) {
      // Empty expert (0 tokens => m/n or variable-K == 0): the GEMM is a no-op
      // and its output is already zeroed by the hipMemset below.
      // hipblasGemmEx rejects zero dims, so skip.
      if (g.M[e] == 0 || g.N[e] == 0 || g.K[e] == 0) continue;
      CHECK_HIPBLAS(hipblasGemmEx(handle, g.opA[e], g.opB[e],
                                  g.M[e], g.N[e], g.K[e], &alpha,
                                  g.Aptr[e], HIPBLAS_R_16B, g.lda[e],
                                  g.Bptr[e], HIPBLAS_R_16B, g.ldb[e], &beta,
                                  g.Cptr[e], HIPBLAS_R_16B, g.ldc[e],
                                  HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
    }
  };

  CHECK_HIP(hipMemset(dC, 0, nC * sizeof(bf16)));
  for (int r = 0; r < 30; r++) launch_loop();
  CHECK_HIP(hipDeviceSynchronize());
  printf("  [hipBLAS per-expert loop  ]\n");
  if (do_verify) { readback_convert(); verify(hCf, hRef, tol); }

  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) launch_loop();
  CHECK_HIP(hipDeviceSynchronize());
  auto t1 = std::chrono::steady_clock::now();
  double loop_us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-3 / repeat;
  double loop_tf = gmm_gflops(mode, batch_sizes, hidden_in, hidden_out, loop_us) / 1000.0;
  printf("  time: %.3f us | %.2f TFLOP/s\n", loop_us, loop_tf);

  // ---------------------------------------------------------------------------
  // Backend 2: single-launch grouped GEMM via the hipBLASLt grouped-gemm
  // extension (hipblaslt_ext::GroupedGemm), the HIP analogue of CUDA's
  // cublasGemmGroupedBatchedEx. Empty experts are dropped from the group list
  // (the grouped API needs positive m/n/k); their outputs stay zeroed by the
  // memset above. All experts in a mode share the same opA/opB, so a single
  // GemmProblemType (set via the constructor) covers the whole group.
  // ---------------------------------------------------------------------------
  std::vector<int64_t> gM, gN, gK, gBatch;
  std::vector<hipblaslt_ext::GemmEpilogue> gEpi;
  std::vector<hipblaslt_ext::GemmInputs> gInputs;
  for (int e = 0; e < E; e++) {
    if (g.M[e] == 0 || g.N[e] == 0 || g.K[e] == 0) continue;
    gM.push_back(g.M[e]); gN.push_back(g.N[e]); gK.push_back(g.K[e]);
    gBatch.push_back(1);
    gEpi.emplace_back();
    hipblaslt_ext::GemmInputs in;
    in.setA(g.Aptr[e]); in.setB(g.Bptr[e]);
    in.setC(g.Cptr[e]); in.setD(g.Cptr[e]);
    in.setAlpha(&alpha); in.setBeta(&beta);
    gInputs.push_back(std::move(in));
  }

  hipblasLtHandle_t lt_handle;
  CHECK_HIPBLAS(hipblasLtCreate(&lt_handle));

  hipblaslt_ext::GroupedGemm gg(lt_handle, g.opA[0], g.opB[0],
                                HIP_R_16BF, HIP_R_16BF, HIP_R_16BF, HIP_R_16BF,
                                HIPBLAS_COMPUTE_32F);
  CHECK_HIPBLAS(gg.setProblem(gM, gN, gK, gBatch, gEpi, gInputs));

  const size_t workspaceSize = 128ULL * 1024 * 1024;
  hipblaslt_ext::GemmPreference pref;
  pref.setMaxWorkspaceBytes(workspaceSize);

  std::vector<hipblasLtMatmulHeuristicResult_t> heuristic;
  hipblasStatus_t hst = gg.algoGetHeuristic(32, pref, heuristic);

  int chosen = -1;
  for (size_t i = 0; hst == HIPBLAS_STATUS_SUCCESS && i < heuristic.size(); i++) {
    size_t ws = 0;
    if (gg.isAlgoSupported(heuristic[i].algo, ws) == HIPBLAS_STATUS_SUCCESS &&
        ws <= workspaceSize) { chosen = (int)i; break; }
  }

  if (chosen < 0) {
    printf("  [hipBLASLt grouped GEMM   ] no supported algorithm for this config -- skipped\n");
  } else {
    void* workspace = nullptr;
    CHECK_HIP(hipMalloc(&workspace, workspaceSize));
    // useUserArgs=false so we can launch with the simple run(stream) entry point.
    CHECK_HIPBLAS(gg.initialize(heuristic[chosen].algo, workspace, false));

    CHECK_HIP(hipMemset(dC, 0, nC * sizeof(bf16)));
    for (int r = 0; r < 30; r++) CHECK_HIPBLAS(gg.run(0));
    CHECK_HIP(hipDeviceSynchronize());
    printf("  [hipBLASLt grouped GEMM   ]\n");
    if (do_verify) { readback_convert(); verify(hCf, hRef, tol); }

    auto s0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; r++) CHECK_HIPBLAS(gg.run(0));
    CHECK_HIP(hipDeviceSynchronize());
    auto s1 = std::chrono::steady_clock::now();
    double grp_us = std::chrono::duration_cast<std::chrono::nanoseconds>(s1 - s0).count() * 1e-3 / repeat;
    double grp_tf = gmm_gflops(mode, batch_sizes, hidden_in, hidden_out, grp_us) / 1000.0;
    printf("  time: %.3f us | %.2f TFLOP/s | grouped/loop speedup: %.2fx\n",
           grp_us, grp_tf, loop_us / grp_us);

    CHECK_HIP(hipFree(workspace));
  }

  CHECK_HIP(hipFree(dA)); CHECK_HIP(hipFree(dB)); CHECK_HIP(hipFree(dC));
  CHECK_HIPBLAS(hipblasDestroy(handle));
  CHECK_HIPBLAS(hipblasLtDestroy(lt_handle));
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
  // Real MoE routing leaves some experts with zero tokens (batch_sizes[e] == 0).
  int empty_experts = 0;
  for (int e = 0; e < num_experts; e++) if (batch_sizes[e] == 0) empty_experts++;

  printf("megablocks grouped GEMM: %d experts, hidden_in=%d, hidden_out=%d, "
         "avg_tokens/expert=%d, repeat=%d (bfloat16); %d empty expert(s)\n",
         num_experts, hidden_in, hidden_out, avg_tokens, repeat, empty_experts);

  run_mode(GMM_FWD,    batch_sizes, hidden_in, hidden_out, repeat);
  run_mode(GMM_FWD_TB, batch_sizes, hidden_in, hidden_out, repeat);
  run_mode(GMM_DW,     batch_sizes, hidden_in, hidden_out, repeat);

  return 0;
}
