// ----------------------------------------------------------------------------
// CPU reference for the Triangle Multiplicative Update (TriMul)
//
// Derived from NVIDIA cuEquivariance:
//   triangle_multiplicative_update() and sigmoid_gated_dual_gemm().
//
// Forward pass (per batch b, with sequence length N and hidden dim D):
//   1. x  = LayerNorm_D(x)                              # input normalization
//   2. a,b = sigmoid(x @ g_in^T + gb) * (x @ p_in^T + pb) * mask   # dual gemm
//            split along the 2D output channels into a and b
//   3. outgoing: t[d,i,j] = sum_k a[d,i,k] * b[d,j,k]   # triangle projection
//      incoming: t[d,i,j] = sum_k a[d,k,i] * b[d,k,j]
//   4. t  = LayerNorm_D(t)                              # output normalization
//   5. out = sigmoid(x_in @ g_out^T + gb) * (t @ p_out^T + pb)     # gated out
//
// Layout (all row-major, float):
//   x, x_norm, x_out, out : (B, N, N, D)
//   mask                  : (B, N, N)
//   a, b, tri             : (B, D, N, N)
//   p_in_w, g_in_w        : (2D, D)      p_in_b, g_in_b : (2D)
//   p_out_w, g_out_w      : (D,  D)      p_out_b, g_out_b: (D)
//   norm_*_w, norm_*_b    : (D)
// ----------------------------------------------------------------------------

#ifndef REFERENCE_H
#define REFERENCE_H

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

// The tensors are indexed with int, so reject sizes whose derived quantities
// would overflow rather than silently running on a corrupt problem.
static bool valid_problem_size(int B, int N, int D, int repeat) {
  if (B <= 0 || N <= 0 || D <= 0 || repeat <= 0) return false;
  if (D > INT_MAX / 2) return false;                    // 2D input channels
  const long long nn = (long long)N * N;
  if (nn > INT_MAX) return false;                       // N * N
  const long long rows = (long long)B * nn;
  if (rows > INT_MAX) return false;                     // rows = B * N * N
  if ((long long)B * D > INT_MAX) return false;         // batched GEMM planes
  // (B, N, N, D) elements, and their size in bytes
  if (rows > (long long)(SIZE_MAX / sizeof(float)) / D) return false;
  return true;
}

static void* xmalloc(size_t bytes) {
  void* p = malloc(bytes);
  if (p == NULL) {
    fprintf(stderr, "Failed to allocate %zu bytes on the host\n", bytes);
    exit(EXIT_FAILURE);
  }
  return p;
}

static inline float sigmoidf_ref(float v) { return 1.0f / (1.0f + expf(-v)); }

// LayerNorm over the last (D) dimension of a single row.
static void layernorm_row_ref(const float* in, float* out, const float* w,
                              const float* b, int D, float eps) {
  float mean = 0.0f;
  for (int d = 0; d < D; d++) mean += in[d];
  mean /= D;
  float var = 0.0f;
  for (int d = 0; d < D; d++) { float s = in[d] - mean; var += s * s; }
  var /= D;
  float rstd = 1.0f / sqrtf(var + eps);
  for (int d = 0; d < D; d++)
    out[d] = (in[d] - mean) * rstd * w[d] + b[d];
}

void trimul_forward_ref(
    float* out,                       // (B, N, N, D)
    const float* x_in,                // (B, N, N, D)
    const float* mask,                // (B, N, N)
    const float* norm_in_w, const float* norm_in_b,
    const float* p_in_w,  const float* p_in_b,   // (2D, D), (2D)
    const float* g_in_w,  const float* g_in_b,
    const float* norm_out_w, const float* norm_out_b,
    const float* p_out_w, const float* p_out_b,  // (D, D), (D)
    const float* g_out_w, const float* g_out_b,
    int B, int N, int D, int outgoing, float eps) {

  const int NN = N * N;
  std::vector<float> xn((size_t)B * NN * D);     // input-normalized (x_in path)
  std::vector<float> a((size_t)B * D * NN);
  std::vector<float> bb((size_t)B * D * NN);
  std::vector<float> tri((size_t)B * D * NN);
  std::vector<float> xout((size_t)B * NN * D);   // output-normalized

  const long long rows = (long long)B * NN;

  // The reference is O(B*D*N^3) in the triangle projection alone, which is
  // minutes of single-threaded work at the sequence lengths AlphaFold uses,
  // so every stage is spread over the cores.  Builds without -fopenmp simply
  // ignore the pragmas and run it serially.

  // 1. input LayerNorm over D
#pragma omp parallel for
  for (long long row = 0; row < rows; row++)
    layernorm_row_ref(x_in + row * D, xn.data() + row * D, norm_in_w, norm_in_b, D, eps);

  // 2. fused sigmoid-gated dual gemm -> a, b  (channels 0..D -> a, D..2D -> b)
#pragma omp parallel for collapse(3)
  for (int b = 0; b < B; b++)
    for (int i = 0; i < N; i++)
      for (int k = 0; k < N; k++) {
        const float* xr = xn.data() + (((size_t)b * N + i) * N + k) * D;
        const float m = mask[((size_t)b * N + i) * N + k];
        for (int c = 0; c < 2 * D; c++) {
          float proj = p_in_b[c], gate = g_in_b[c];
          const float* pw = p_in_w + (size_t)c * D;
          const float* gw = g_in_w + (size_t)c * D;
          for (int d = 0; d < D; d++) { proj += xr[d] * pw[d]; gate += xr[d] * gw[d]; }
          float v = sigmoidf_ref(gate) * proj * m;
          int d = c < D ? c : c - D;
          float* dst = (c < D ? a.data() : bb.data());
          dst[(((size_t)b * D + d) * N + i) * N + k] = v;
        }
      }

  // 3. triangle projection (batched over b, d)
#pragma omp parallel for collapse(2)
  for (int b = 0; b < B; b++)
    for (int d = 0; d < D; d++) {
      const float* ad = a.data()  + ((size_t)b * D + d) * NN;
      const float* bd = bb.data() + ((size_t)b * D + d) * NN;
      float* td = tri.data() + ((size_t)b * D + d) * NN;
      for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
          float acc = 0.0f;
          if (outgoing)
            for (int k = 0; k < N; k++) acc += ad[i * N + k] * bd[j * N + k];
          else
            for (int k = 0; k < N; k++) acc += ad[k * N + i] * bd[k * N + j];
          td[i * N + j] = acc;
        }
    }

  // 4. output LayerNorm over D  (gather strided channel, layout d,i,j -> i,j,d)
  std::vector<float> tmp(D);   // firstprivate: one gather buffer per thread
#pragma omp parallel for collapse(3) firstprivate(tmp)
  for (int b = 0; b < B; b++)
    for (int i = 0; i < N; i++)
      for (int j = 0; j < N; j++) {
        for (int d = 0; d < D; d++)
          tmp[d] = tri[(((size_t)b * D + d) * N + i) * N + j];
        layernorm_row_ref(tmp.data(),
                          xout.data() + (((size_t)b * N + i) * N + j) * D,
                          norm_out_w, norm_out_b, D, eps);
      }

  // 5. output gating: sigmoid(x_in @ g_out) * (x_out @ p_out)
#pragma omp parallel for
  for (long long row = 0; row < rows; row++) {
    const float* xir = xn.data()   + row * D;
    const float* xor_ = xout.data() + row * D;
    float* orow = out + row * D;
    for (int dp = 0; dp < D; dp++) {
      float gate = g_out_b[dp], proj = p_out_b[dp];
      const float* gw = g_out_w + (size_t)dp * D;
      const float* pw = p_out_w + (size_t)dp * D;
      for (int d = 0; d < D; d++) { gate += xir[d] * gw[d]; proj += xor_[d] * pw[d]; }
      orow[dp] = sigmoidf_ref(gate) * proj;
    }
  }
}

#endif
