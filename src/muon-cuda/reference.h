#ifndef REFERENCE_H
#define REFERENCE_H

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <algorithm>

// Newton-Schulz quintic iteration coefficients (see KellerJordan/Muon)
#define NS_A  3.4445f
#define NS_B -4.7750f
#define NS_C  2.0315f
#define NS_STEPS 5

// Round a float to the nearest bfloat16 value (round-to-nearest-even) and
// return it as a float. This emulates bf16 storage on the host so the CPU
// reference matches the bf16 CUDA kernels (which accumulate in fp32 and store
// bf16). The Newton-Schulz orthogonalization runs in bf16, as in the reference
// implementation https://github.com/KellerJordan/Muon .
static inline float bf16r(float f)
{
  uint32_t x;
  memcpy(&x, &f, sizeof(x));
  if ((x & 0x7fffffffu) > 0x7f800000u) return f; // NaN passthrough
  const uint32_t lsb = (x >> 16) & 1u;
  x += 0x7fffu + lsb;                             // round to nearest even
  x &= 0xffff0000u;
  float r;
  memcpy(&r, &x, sizeof(r));
  return r;
}

// C[M x N] = A[M x K] @ B[K x N]  (bf16 storage, fp32 accumulation)
static void ref_mm(const float* A, const float* B, float* C,
                   int M, int N, int K)
{
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      float s = 0.f;
      #pragma omp parallel for reduction(+:s)
      for (int k = 0; k < K; k++)
        s += A[i * K + k] * B[k * N + j];
      C[i * N + j] = bf16r(s);
    }
}

// C[M x N] = A[M x K] @ B^T, where B is [N x K]  (bf16 storage, fp32 accum)
static void ref_mm_abt(const float* A, const float* B, float* C,
                       int M, int N, int K)
{
  #pragma omp parallel for collapse(2)
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      float s = 0.f;
      #pragma omp parallel for reduction(+:s)
      for (int k = 0; k < K; k++)
        s += A[i * K + k] * B[j * K + k];
      C[i * N + j] = bf16r(s);
    }
}

// Newton-Schulz orthogonalization of an m x n matrix X (with m <= n), in bf16.
static void ref_newton_schulz(float* X, int m, int n)
{
  double ss = 0.0;
  for (int i = 0; i < m * n; i++) ss += (double)X[i] * (double)X[i];
  float inv = 1.f / ((float)sqrt(ss) + 1e-7f);
  for (int i = 0; i < m * n; i++) X[i] = bf16r(X[i] * inv);

  float* A  = (float*) malloc (sizeof(float) * m * m);
  float* A2 = (float*) malloc (sizeof(float) * m * m);
  float* B  = (float*) malloc (sizeof(float) * m * m);
  float* BX = (float*) malloc (sizeof(float) * m * n);

  for (int s = 0; s < NS_STEPS; s++) {
    ref_mm_abt(X, X, A, m, m, n);   // A  = X @ X^T
    ref_mm(A, A, A2, m, m, m);      // A2 = A @ A
    for (int i = 0; i < m * m; i++)
      B[i] = bf16r(NS_B * A[i] + NS_C * A2[i]);
    ref_mm(B, X, BX, m, n, m);      // BX = B @ X
    for (int i = 0; i < m * n; i++)
      X[i] = bf16r(NS_A * X[i] + BX[i]);
  }

  free(A); free(A2); free(B); free(BX);
}

// One Muon update step on a P x Q parameter matrix p, with momentum buffer mom
// and gradient g. Uses nesterov momentum (as in the reference implementation).
static void ref_muon_update(float* p, float* mom, const float* g,
                            int P, int Q, float beta, float lr, float wd)
{
  const int N = P * Q;
  float* update = (float*) malloc (sizeof(float) * N);

  // momentum.lerp_(grad, 1-beta); update = grad.lerp_(momentum, beta)  (fp32)
  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    float mi = beta * mom[i] + (1.f - beta) * g[i];
    mom[i] = mi;
    update[i] = (1.f - beta) * g[i] + beta * mi;
  }

  const int mm = std::min(P, Q);
  const int nn = std::max(P, Q);
  float* X = (float*) malloc (sizeof(float) * N);

  // convert update to bf16 working matrix, transposing if P > Q (rows <= cols)
  if (P > Q) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < P; i++)
      for (int j = 0; j < Q; j++)
        X[j * P + i] = bf16r(update[i * Q + j]);
  } else {
    for (int i = 0; i < N; i++) X[i] = bf16r(update[i]);
  }

  ref_newton_schulz(X, mm, nn);

  if (P > Q) {
    // transpose X (Q x P) back to update (P x Q)
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < Q; i++)
      for (int j = 0; j < P; j++)
        update[j * Q + i] = X[i * P + j];
  } else {
    memcpy(update, X, sizeof(float) * N);
  }

  // update *= max(1, rows/cols)^0.5  (in bf16)
  float scale = sqrtf(fmaxf(1.f, (float)P / (float)Q));
  for (int i = 0; i < N; i++) update[i] = bf16r(update[i] * scale);

  // p *= (1 - lr*wd); p += -lr * update
  for (int i = 0; i < N; i++)
    p[i] = p[i] * (1.f - lr * wd) - lr * update[i];

  free(update);
  free(X);
}

static void reference(int repeat, float* p, float* mom, const float* g,
                      int P, int Q, float beta, float lr, float wd)
{
  for (int i = 0; i < repeat; i++)
    ref_muon_update(p, mom, g, P, Q, beta, lr, wd);
}

#endif
