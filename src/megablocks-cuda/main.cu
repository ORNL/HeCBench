// megablocks benchmark.
//
// Tensors are bfloat16, row-major, alpha=1, beta=0

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "reference.h"

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s at %s:%d\n",                            \
              cudaGetErrorString(err), __FILE__, __LINE__);                  \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

#define CHECK_CUBLAS(call)                                                   \
  do {                                                                       \
    cublasStatus_t st = (call);                                             \
    if (st != CUBLAS_STATUS_SUCCESS) {                                       \
      fprintf(stderr, "cuBLAS error %d at %s:%d\n", st, __FILE__, __LINE__); \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

using bf16 = __nv_bfloat16;

// Per-group cuBLAS parameters after megablocks' row-major operand swap:
// row-major C = op(a)*op(b) is computed by cublas as column-major with the A/B
// operands swapped. So the grouped-batched "A" operand is megablocks' b, and its
// "B" operand is megablocks' a.
struct Groups {
  int E;
  std::vector<cublasOperation_t> opA, opB;  // opA applies to Aptr(=b), opB to Bptr(=a)
  std::vector<int> M, N, K, lda, ldb, ldc;
  std::vector<const void*> Aptr;            // -> b blocks
  std::vector<const void*> Bptr;            // -> a blocks
  std::vector<void*> Cptr;                  // -> c blocks
};

static const char* mode_name(GmmMode m) {
  switch (m) {
    case GMM_FWD:    return "fwd  (trans_a=0, trans_b=0)";
    case GMM_FWD_TB: return "fwd  (trans_a=0, trans_b=1)";
    default:         return "dw   (trans_a=1, variable K)";
  }
}

// Build the per-group parameters and device pointers for a given mode.
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
      // megablocks: m=N, k=K, n=m_e; lda(a)=K, ldb(b)= (tb? K : N); ta=N.
      g.opA[e] = tb ? CUBLAS_OP_T : CUBLAS_OP_N;   // op on b
      g.opB[e] = CUBLAS_OP_N;                       // op on a
      g.M[e]   = N;
      g.N[e]   = m_e;
      g.K[e]   = K;
      g.lda[e] = tb ? K : N;                        // ldb-of-b
      g.ldb[e] = K;                                 // lda-of-a
      g.ldc[e] = N;
      g.Aptr[e] = dB + offB;                        // b block
      g.Bptr[e] = dA + offA;                        // a block
      g.Cptr[e] = dC + offC;
      offA += (size_t)m_e * K;
      offB += (size_t)K * N;                        // weight block (K*N == N*K)
      offC += (size_t)m_e * N;
    } else { // GMM_DW: variable K; per expert C_e[M x N] = A_e[k_e x M]^T * B_e[k_e x N]
      const int k_e = batch_sizes[e];
      const int M = hidden_in;
      // megablocks: m=N, k=k_e, n=M; lda(a)=M, ldb(b)=N; ta=T, tb=N.
      g.opA[e] = CUBLAS_OP_N;   // op on b
      g.opB[e] = CUBLAS_OP_T;   // op on a
      g.M[e]   = N;
      g.N[e]   = M;
      g.K[e]   = k_e;
      g.lda[e] = N;             // ldb-of-b
      g.ldb[e] = M;             // lda-of-a
      g.ldc[e] = N;
      g.Aptr[e] = dB + offB;    // b block (k_e x N)
      g.Bptr[e] = dA + offA;    // a block (k_e x M)
      g.Cptr[e] = dC + offC;    // c block (M x N)
      offA += (size_t)k_e * M;
      offB += (size_t)k_e * N;
      offC += (size_t)M * N;
    }
  }

  // Variable-K (dw) only: megablocks sorts the grouped problems by descending K
  // (std::stable_sort on problem.k()) before launching. Replicate that ordering
  // here so the grouped GEMM sees experts in the same order. Each group carries
  // its own pointers/leading dimensions, so reordering the whole tuple keeps the
  // outputs landing in their original device blocks (verification is unaffected).
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
  //const int N = hidden_out;

  // Determine buffer element counts for the flat row-major tensors.
  size_t nA = 0, nB = 0, nC = 0;
  for (int e = 0; e < E; e++) {
    if (mode == GMM_FWD || mode == GMM_FWD_TB) {
      nA += (size_t)batch_sizes[e] * hidden_in;
      nB += (size_t)hidden_in * hidden_out;
      nC += (size_t)batch_sizes[e] * hidden_out;
    } else {
      nA += (size_t)batch_sizes[e] * hidden_in;   // hidden_in == M
      nB += (size_t)batch_sizes[e] * hidden_out;
      nC += (size_t)hidden_in * hidden_out;
    }
  }

  printf("---- mode: %s ----\n", mode_name(mode));

  // Host float inputs and bf16 copies.
  std::vector<float> hAf(nA), hBf(nB), hRef(nC, 0.f);
  std::vector<bf16>  hA(nA), hB(nB);
  srand48(123);
  for (size_t i = 0; i < nA; i++) { hAf[i] = (float)drand48() - 0.5f; hA[i] = __float2bfloat16(hAf[i]); }
  for (size_t i = 0; i < nB; i++) { hBf[i] = (float)drand48() - 0.5f; hB[i] = __float2bfloat16(hBf[i]); }

  bf16 *dA, *dB, *dC;
  CHECK_CUDA(cudaMalloc(&dA, nA * sizeof(bf16)));
  CHECK_CUDA(cudaMalloc(&dB, nB * sizeof(bf16)));
  CHECK_CUDA(cudaMalloc(&dC, nC * sizeof(bf16)));
  CHECK_CUDA(cudaMemcpy(dA, hA.data(), nA * sizeof(bf16), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB, hB.data(), nB * sizeof(bf16), cudaMemcpyHostToDevice));

  Groups g = buildGroups(mode, batch_sizes, hidden_in, hidden_out, dA, dB, dC);

  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  const float alpha = 1.f, beta = 0.f;

  // ---- CPU reference ----
  // The O(sum m*n*k) host reference is only tractable for modest sizes; skip it
  // for large problems (correctness is validated on smaller shapes).
  double total_flop = gmm_gflops(mode, batch_sizes, hidden_in, hidden_out, 1.0) * 1e3; // == 2*sum(m*n*k)
  const bool do_verify = total_flop < 2e10;
  const double tol = 1.0;  // bf16 accumulation into fp32; generous absolute tol
  std::vector<float> hCf(nC);
  if (do_verify)
    gmm_ref(mode, batch_sizes, hidden_in, hidden_out, hAf, hBf, hRef);
  else
    printf("  (host verification skipped: problem too large)\n");

  auto readback_convert = [&]() {
    std::vector<bf16> hC(nC);
    CHECK_CUDA(cudaMemcpy(hC.data(), dC, nC * sizeof(bf16), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < nC; i++) hCf[i] = __bfloat162float(hC[i]);
  };

  // ---------------------------------------------------------------------------
  // Backend 1: per-expert cublasGemmEx loop
  // ---------------------------------------------------------------------------
  auto launch_loop = [&]() {
    for (int e = 0; e < E; e++) {
      // Empty expert (0 tokens => m/n or variable-K == 0): the GEMM is a no-op
      // and its output is already zeroed by the cudaMemset below.
      // cublasGemmEx rejects zero dims, so skip.
      if (g.M[e] == 0 || g.N[e] == 0 || g.K[e] == 0) continue;
      CHECK_CUBLAS(cublasGemmEx(handle, g.opA[e], g.opB[e],
                                g.M[e], g.N[e], g.K[e], &alpha,
                                g.Aptr[e], CUDA_R_16BF, g.lda[e],
                                g.Bptr[e], CUDA_R_16BF, g.ldb[e], &beta,
                                g.Cptr[e], CUDA_R_16BF, g.ldc[e],
                                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
  };

  CHECK_CUDA(cudaMemset(dC, 0, nC * sizeof(bf16)));
  for (int r = 0; r < 30; r++) launch_loop();
  CHECK_CUDA(cudaDeviceSynchronize());
  printf("  [cuBLAS per-expert loop  ]\n");
  if (do_verify) { readback_convert(); verify(hCf, hRef, tol); }

  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) launch_loop();
  CHECK_CUDA(cudaDeviceSynchronize());
  auto t1 = std::chrono::steady_clock::now();
  double loop_us = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() * 1e-3 / repeat;
  double loop_tf = gmm_gflops(mode, batch_sizes, hidden_in, hidden_out, loop_us) / 1000.0;
  printf("  time: %.3f us | %.2f TFLOP/s\n", loop_us, loop_tf);

  // ---------------------------------------------------------------------------
  // Backend 2: single-launch cublasGemmGroupedBatchedEx (CUDA 12.5+).
  // Pointer arrays must live in device memory; sizes/ld/ops arrays on host.
  // Empty experts are dropped from the group list (the grouped-batched API
  // requires positive m/n/k); their outputs stay zeroed by the memset above
  // ---------------------------------------------------------------------------
  std::vector<cublasOperation_t> fopA, fopB;
  std::vector<int> fM, fN, fK, flda, fldb, fldc;
  std::vector<const void*> fAptr, fBptr; std::vector<void*> fCptr;
  for (int e = 0; e < E; e++) {
    if (g.M[e] == 0 || g.N[e] == 0 || g.K[e] == 0) continue;
    fopA.push_back(g.opA[e]);   fopB.push_back(g.opB[e]);
    fM.push_back(g.M[e]);       fN.push_back(g.N[e]);       fK.push_back(g.K[e]);
    flda.push_back(g.lda[e]);   fldb.push_back(g.ldb[e]);   fldc.push_back(g.ldc[e]);
    fAptr.push_back(g.Aptr[e]); fBptr.push_back(g.Bptr[e]); fCptr.push_back(g.Cptr[e]);
  }
  const int Eg = (int)fM.size();

  const void **dAarr, **dBarr; void **dCarr;
  CHECK_CUDA(cudaMalloc(&dAarr, Eg * sizeof(void*)));
  CHECK_CUDA(cudaMalloc(&dBarr, Eg * sizeof(void*)));
  CHECK_CUDA(cudaMalloc(&dCarr, Eg * sizeof(void*)));
  CHECK_CUDA(cudaMemcpy(dAarr, fAptr.data(), Eg * sizeof(void*), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dBarr, fBptr.data(), Eg * sizeof(void*), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dCarr, fCptr.data(), Eg * sizeof(void*), cudaMemcpyHostToDevice));

  std::vector<float> alpha_arr(Eg, 1.f), beta_arr(Eg, 0.f);
  std::vector<int> group_size(Eg, 1);

  auto launch_grouped = [&]() {
    return cublasGemmGroupedBatchedEx(handle, fopA.data(), fopB.data(),
                                      fM.data(), fN.data(), fK.data(),
                                      alpha_arr.data(),
                                      dAarr, CUDA_R_16BF, flda.data(),
                                      dBarr, CUDA_R_16BF, fldb.data(),
                                      beta_arr.data(),
                                      dCarr, CUDA_R_16BF, fldc.data(),
                                      Eg, group_size.data(),
                                      CUBLAS_COMPUTE_32F);
  };

  CHECK_CUDA(cudaMemset(dC, 0, nC * sizeof(bf16)));
  cublasStatus_t gstat = launch_grouped();
  if (gstat != CUBLAS_STATUS_SUCCESS) {
    printf("  [cublasGemmGroupedBatchedEx] not supported for this config (status %d) -- skipped\n", gstat);
  } else {
    for (int r = 0; r < 30; r++) CHECK_CUBLAS(launch_grouped());
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("  [cublasGemmGroupedBatchedEx]\n");
    if (do_verify) { readback_convert(); verify(hCf, hRef, tol); }

    auto s0 = std::chrono::steady_clock::now();
    for (int r = 0; r < repeat; r++) CHECK_CUBLAS(launch_grouped());
    CHECK_CUDA(cudaDeviceSynchronize());
    auto s1 = std::chrono::steady_clock::now();
    double grp_us = std::chrono::duration_cast<std::chrono::nanoseconds>(s1 - s0).count() * 1e-3 / repeat;
    double grp_tf = gmm_gflops(mode, batch_sizes, hidden_in, hidden_out, grp_us) / 1000.0;
    printf("  time: %.3f us | %.2f TFLOP/s | grouped/loop speedup: %.2fx\n",
           grp_us, grp_tf, loop_us / grp_us);
  }

  CHECK_CUDA(cudaFree(dAarr)); CHECK_CUDA(cudaFree(dBarr)); CHECK_CUDA(cudaFree(dCarr));
  CHECK_CUDA(cudaFree(dA)); CHECK_CUDA(cudaFree(dB)); CHECK_CUDA(cudaFree(dC));
  CHECK_CUBLAS(cublasDestroy(handle));
}

int main(int argc, char* argv[])
{
  // Usage: main [repeat] [num_experts] [hidden_in] [hidden_out] [avg_tokens]
  const int repeat      = (argc > 1) ? atoi(argv[1]) : 100;
  const int num_experts = (argc > 2) ? atoi(argv[2]) : 64;
  const int hidden_in   = (argc > 3) ? atoi(argv[3]) : 2048;  // K (fwd) / M (dw)
  const int hidden_out  = (argc > 4) ? atoi(argv[4]) : 2048;  // N
  const int avg_tokens  = (argc > 5) ? atoi(argv[5]) : 16;

  // Skewed router assignment: draw a per-expert weight, then assign tokens so the
  // load is imbalanced (as in real MoE). Every expert gets at least one token.
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

  run_mode(GMM_FWD,    batch_sizes, hidden_in, hidden_out, repeat);
  run_mode(GMM_FWD_TB, batch_sizes, hidden_in, hidden_out, repeat);
  run_mode(GMM_DW,     batch_sizes, hidden_in, hidden_out, repeat);

  return 0;
}
