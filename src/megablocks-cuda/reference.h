#ifndef REFERENCE_H
#define REFERENCE_H

#include <vector>
#include <cmath>
#include <cstdio>

// Host reference for the megablocks grouped GEMM ("gmm") op.
//
// megablocks stores all tensors row-major (bfloat16 on device). A router sends a
// variable number of tokens to each of E experts, so batch_sizes[e] gives the
// per-expert group size. Three modes are used by the MoE MLP (see grouped_gemm.cu):
//
//   fwd     (trans_a=false, trans_b=false):
//       a = (tokens, hidden_in),  b = (E, hidden_in, hidden_out),  c = (tokens, hidden_out)
//       per expert e:  C_e[m_e x N] = A_e[m_e x K] * B_e[K x N]        (m_e = batch_sizes[e])
//
//   fwd_tb  (trans_a=false, trans_b=true):
//       a = (tokens, hidden_in),  b = (E, hidden_out, hidden_in),  c = (tokens, hidden_out)
//       per expert e:  C_e[m_e x N] = A_e[m_e x K] * B_e[N x K]^T
//
//   dw      (trans_a=true, variable K, backward weight gradient):
//       a = (tokens, hidden_in),  b = (tokens, hidden_out),  c = (E, hidden_in, hidden_out)
//       per expert e:  C_e[M x N] = A_e[k_e x M]^T * B_e[k_e x N]       (k_e = batch_sizes[e])
//
// All references are computed in float; the device output (bf16) is converted
// back to float before comparison. The reference always uses alpha=1, beta=0.

enum GmmMode { GMM_FWD = 0, GMM_FWD_TB = 1, GMM_DW = 2 };

// Row-major reference. Inputs A, B are flat float buffers laid out exactly like
// the device tensors; C is the flat float output.
inline void gmm_ref(GmmMode mode,
                    const std::vector<int> &batch_sizes,
                    int hidden_in,   // K in fwd, M(=rows of each C block) in dw
                    int hidden_out,  // N
                    const std::vector<float> &A,
                    const std::vector<float> &B,
                    std::vector<float> &C)
{
  const int E = (int)batch_sizes.size();
  const int N = hidden_out;

  if (mode == GMM_FWD || mode == GMM_FWD_TB) {
    const int K = hidden_in;
    size_t offA = 0, offB = 0, offC = 0;
    const size_t Bstride = (size_t)K * N;  // each expert weight block
    for (int e = 0; e < E; e++) {
      const int M = batch_sizes[e];
      const float *Ae = A.data() + offA;
      const float *Be = B.data() + offB;
      float *Ce = C.data() + offC;
      for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
          float acc = 0.f;
          for (int p = 0; p < K; p++) {
            float a = Ae[(size_t)i * K + p];
            // B_e row-major: fwd -> [p, j] (K x N); fwd_tb -> [j, p] (N x K)
            float b = (mode == GMM_FWD) ? Be[(size_t)p * N + j]
                                        : Be[(size_t)j * K + p];
            acc += a * b;
          }
          Ce[(size_t)i * N + j] = acc;
        }
      }
      offA += (size_t)M * K;
      offB += Bstride;
      offC += (size_t)M * N;
    }
  } else { // GMM_DW: variable K, per expert C_e[M x N] = A_e[k_e x M]^T * B_e[k_e x N]
    const int M = hidden_in;
    size_t offA = 0, offB = 0, offC = 0;
    const size_t Cstride = (size_t)M * N;
    for (int e = 0; e < E; e++) {
      const int Ke = batch_sizes[e];
      const float *Ae = A.data() + offA;  // (k_e x M) row-major
      const float *Be = B.data() + offB;  // (k_e x N) row-major
      float *Ce = C.data() + offC;        // (M x N) row-major
      for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
          float acc = 0.f;
          for (int p = 0; p < Ke; p++)
            acc += Ae[(size_t)p * M + i] * Be[(size_t)p * N + j];
          Ce[(size_t)i * N + j] = acc;
        }
      }
      offA += (size_t)Ke * M;
      offB += (size_t)Ke * N;
      offC += Cstride;
    }
  }
}

inline bool verify(const std::vector<float> &test,
                   const std::vector<float> &ref,
                   double tol)
{
  double max_err = 0.0;
  for (size_t i = 0; i < ref.size(); i++)
    max_err = std::fmax(max_err, std::fabs((double)test[i] - (double)ref[i]));
  bool ok = max_err <= tol;
  printf("  Maximum absolute error: %e (tolerance %e) -> %s\n",
         max_err, tol, ok ? "PASS" : "FAIL");
  return ok;
}

// Aggregate FLOPs across all groups: 2*M*N*K per group.
inline double gmm_gflops(GmmMode mode, const std::vector<int> &batch_sizes,
                         int hidden_in, int hidden_out, double avg_time_us)
{
  double flop = 0.0;
  const int N = hidden_out;
  if (mode == GMM_DW) {
    const int M = hidden_in;
    for (int b : batch_sizes) flop += 2.0 * M * N * b; // K = b
  } else {
    const int K = hidden_in;
    for (int b : batch_sizes) flop += 2.0 * b * N * K; // M = b
  }
  return flop / (avg_time_us * 1e3); // GFLOP/s
}

#endif
