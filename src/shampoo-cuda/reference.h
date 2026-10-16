#ifndef SHAMPOO_REFERENCE_H
#define SHAMPOO_REFERENCE_H

// Host reference for the order-2 root-inverse Shampoo step in
// distributed_shampoo/preconditioner/shampoo_preconditioner_list.py
// (facebookresearch/optimizers main, d24a1491797e5bf2c2985d8789f3a28293d87a6f).
//
// Adam-style Kronecker factors (beta2 = 0.999, weighting_factor = 1 - beta2,
// bias correction on), no dimension normalization. An order-2 block uses
// inverse exponent 1/4, so each factor's amortized computation is the coupled
// Newton-Schulz inverse fourth root from matrix_functions.py
// (NewtonSchulzRootInvConfig: relative_epsilon 1e-6, Polar Express schedule).
// _precondition_grad contracts dimension 0 of the gradient with dimension 0 of
// each factor, which for these symmetric factors is out = Linv^T @ G @ Rinv.
//
// That direction is then rescaled by AdaGrad grafting, Section 2.4 and
// Algorithm 1 of Shi et al., arXiv:2309.06497. The grafted state is the sum
// of squared gradient entries. Its search direction is G / (sqrt(A) + eps),
// with eps = 1e-10 from AdaGradPreconditionerConfig in the same commit.
// The Shampoo direction is multiplied by ||P_graft||_F / (||P_shampoo||_F + 1e-16).
// The 1e-16 addend is the zero-direction guard in _precondition_and_grafting.

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <omp.h>

static const float kBeta2 = 0.999f;
static const float kWeight = 1.f - kBeta2;
static const float kEpsilon = 1e-12f;
static const float kRelativeEpsilon = 1e-6f;
// AdaGrad grafting denominator (AdaGradPreconditionerConfig.epsilon).
static const float kGraftEpsilon = 1e-10f;
// Added to ||P_shampoo||_F before the grafting ratio, matching
// torch._foreach_add_(shampoo_norm_list, 1e-16) in _precondition_and_grafting.
static const float kGraftNormEps = 1e-16f;
static const int kShampooRoot = 4;
static const int kNsIters = 10;

// (a, b, c) for p(x) = a x + b x^3 + c x^5. The last rows sum to 1, so the
// scalar map has a fixed point at 1 (inverse square root). Muon's quintic
// uses a different triple and orthogonalizes a rectangular matrix.
static const float kNs[kNsIters][3] = {
    {8.28721201814563f, -23.595886519098837f, 17.300387312530933f},
    {4.107059111542203f, -2.9478499167379106f, 0.5448431082926601f},
    {3.9486908534822946f, -2.9089021159629490f, 0.5518191394370137f},
    {3.3184196573706015f, -2.4884880243148740f, 0.5100489401237200f},
    {2.300652019954817f, -1.6689039845747493f, 0.4188073119525673f},
    {1.891301407787398f, -1.2679958271945868f, 0.3768040894852483f},
    {1.8750014808534479f, -1.2500016453999487f, 0.3750001645474248f},
    {1.875f, -1.25f, 0.375f},
    {1.875f, -1.25f, 0.375f},
    {1.875f, -1.25f, 0.375f},
};

// 1 - beta2^step, with step counted from 1. L and R start at zero, so the
// raw EMA is biased low by this factor. The step divides L and R by it
// before the inverse root. For a gradient that never changes, that division
// makes the corrected factors equal GG^T and G^T G on every step.
static inline float shampoo_bias_correction2(int step) {
  return 1.f - powf(kBeta2, (float)step);
}

// Threads for one reference call. Every loop is workshared across this team.
// The team grows with the factor until it hits the process limit. Past about
// four rows per thread the barriers cost more than the products, so a small
// factor does not start one thread per core.
static int shampoo_threads(int rows) {
  int t = omp_get_max_threads();
  if (t < 1 || rows < 1) return 1;
  long long flops = 2LL * rows * (long long)rows * rows;
  int by_flops = (int)(flops / 50000LL);
  if (by_flops < 1) by_flops = 1;
  int by_rows = rows / 4;
  if (by_rows < 1) by_rows = 1;
  int cap = by_flops < by_rows ? by_flops : by_rows;
  if (cap > rows) cap = rows;
  if (t > cap) t = cap;
  return t;
}

// Row-major C(M x N) = A(M x K) @ B(K x N).
// Called by every thread in ref_shampoo. The outer loop is workshared and
// the contiguous update is simd, so the whole product is parallel without
// an architecture-specific kernel.
static void gemm_nn(const float* A, const float* B, float* C, int M, int N, int K) {
#pragma omp for schedule(static)
  for (int i = 0; i < M; i++) {
    float* crow = C + (size_t)i * N;
#pragma omp simd
    for (int j = 0; j < N; j++) crow[j] = 0.f;
    const float* arow = A + (size_t)i * K;
    for (int k = 0; k < K; k++) {
      float a = arow[k];
      const float* brow = B + (size_t)k * N;
#pragma omp simd
      for (int j = 0; j < N; j++) crow[j] += a * brow[j];
    }
  }
}

// Row-major C(M x N) = A(K x M)^T @ B(K x N).
// The precondition calls this as Linv^T @ G. Linv is symmetric, so the
// transpose is the index convention for that contraction, and the values
// of Linv are unchanged by it. Same worksharing rules as gemm_nn.
static void gemm_tn(const float* A, const float* B, float* C, int M, int N, int K) {
#pragma omp for schedule(static)
  for (int i = 0; i < M; i++) {
    float* crow = C + (size_t)i * N;
#pragma omp simd
    for (int j = 0; j < N; j++) crow[j] = 0.f;
    for (int k = 0; k < K; k++) {
      float a = A[(size_t)k * M + i];
      const float* brow = B + (size_t)k * N;
#pragma omp simd
      for (int j = 0; j < N; j++) crow[j] += a * brow[j];
    }
  }
}

// Row-major C(M x M) = G(M x N) @ G^T, the left Kronecker factor's new sample.
// C(i, j) is the dot product of gradient row i with gradient row j: how
// strongly those two outputs were pushed together on this batch.
static void gram_left(const float* G, float* C, int M, int N) {
#pragma omp for collapse(2) schedule(static)
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < M; j++) {
      float s = 0.f;
      const float* gi = G + (size_t)i * N;
      const float* gj = G + (size_t)j * N;
#pragma omp simd reduction(+:s)
      for (int k = 0; k < N; k++) s += gi[k] * gj[k];
      C[(size_t)i * M + j] = s;
    }
  }
}

// Row-major C(N x N) = G(M x N)^T @ G, the right Kronecker factor's new sample.
// C(i, j) is the dot product of gradient column i with gradient column j:
// how strongly those two inputs were pushed together on this batch.
static void gram_right(const float* G, float* C, int M, int N) {
  // C = G^T @ G, workshared the same way as the other products.
  gemm_tn(G, G, C, N, N, M);
}

// Sum of squares of `count` entries. ||A||_F is the square root of this.
// The inverse root divides by ||A||_F so every singular value is at most 1,
// the range the Polar Express coefficients assume, and the ridge is
// ||A||_F times the relative epsilon.
// `partial` is a shared strip, one cache line per thread (stride 8 doubles).
// A reduction variable has to be shared, and a temporary in this function is not.
static const int kPartialStride = 8;
static double frobenius_sq(const float* A, int count, double* partial) {
  double sum = 0.0;
#pragma omp for schedule(static)
  for (int i = 0; i < count; i++) sum += (double)A[i] * (double)A[i];
  partial[omp_get_thread_num() * kPartialStride] = sum;
#pragma omp barrier
  int nt = omp_get_num_threads();
  double total = 0.0;
  for (int t = 0; t < nt; t++) total += partial[t * kPartialStride];
  // Keep the slots intact until every thread has read them.
#pragma omp barrier
  return total;
}

// Overwrite the n x n matrix A with A^{-1/root}. `root` is a power of two;
// Shampoo passes 4, so the result is the inverse fourth root. `work` holds
// 4 * n * n floats, used as Y, Z, T, and B.
//
// A diagonal ridge max(||A||_F * 1e-6, 1e-12) is added first. A Gram matrix
// of a rectangular gradient is singular, and the ridge keeps the inverse
// root finite in directions the gradient has not spanned.
//
// Each pass normalizes A by ||A||_F into Y, sets Z = I, and runs the coupled
// quintic. One quintic with coefficients (a, b, c) is
//   T = Z Y,  B = b T + c T^2,  Y = a Y + Y B,  Z = a Z + B Z.
// Y is driven toward the square root of the normalized matrix and Z toward
// its inverse square root. An intermediate pass writes
// A = sqrt(||A||_F) * Y, which restores A^{1/2}. The last pass writes
// A = Z / sqrt(||A||_F), which restores A^{-1/2} of the matrix that entered
// that pass. Root 4 is two passes, so the second turns A^{1/2} into A^{-1/4}.
static void ref_inverse_root(float* A, int n, int root, float* work,
                             double* partial) {
  const int count = n * n;
  float* Y = work;
  float* Z = work + count;
  float* T = work + 2 * count;
  float* B = work + 3 * count;

  double sumsq = frobenius_sq(A, count, partial);
  float ridge = fmaxf((float)sqrt(sumsq) * kRelativeEpsilon, kEpsilon);
#pragma omp for schedule(static)
  for (int i = 0; i < n; i++) A[(size_t)i * n + i] += ridge;

  int passes = 0;
  for (int r = root; r > 1; r >>= 1) passes++;

  for (int pass = passes; pass >= 1; pass--) {
    sumsq = frobenius_sq(A, count, partial);
    float alpha = fmaxf((float)sqrt(sumsq), 1e-8f);
    float inv_alpha = 1.f / alpha;
#pragma omp for simd schedule(static)
    for (int i = 0; i < count; i++) Y[i] = A[i] * inv_alpha;
#pragma omp for simd schedule(static)
    for (int i = 0; i < count; i++) Z[i] = 0.f;
#pragma omp for schedule(static)
    for (int i = 0; i < n; i++) Z[(size_t)i * n + i] = 1.f;

    for (int s = 0; s < kNsIters; s++) {
      float a = kNs[s][0], b = kNs[s][1], c = kNs[s][2];
      gemm_nn(Z, Y, T, n, n, n);
      gemm_nn(T, T, B, n, n, n);
#pragma omp for simd schedule(static)
      for (int i = 0; i < count; i++) B[i] = c * B[i] + b * T[i];
      gemm_nn(Y, B, T, n, n, n);
#pragma omp for simd schedule(static)
      for (int i = 0; i < count; i++) Y[i] = a * Y[i] + T[i];
      gemm_nn(B, Z, T, n, n, n);
#pragma omp for simd schedule(static)
      for (int i = 0; i < count; i++) Z[i] = a * Z[i] + T[i];
    }

    float scale = sqrtf(alpha);
    const float* src = (pass > 1) ? Y : Z;
    if (pass > 1) {
#pragma omp for simd schedule(static)
      for (int i = 0; i < count; i++) A[i] = src[i] * scale;
    } else {
      float inv_scale = 1.f / scale;
#pragma omp for simd schedule(static)
      for (int i = 0; i < count; i++) A[i] = src[i] * inv_scale;
    }
  }
}

// Verification gradient for step s, counted from 1. H == nullptr keeps G0,
// which is the timed benchmark. A second pattern makes each Gram matrix a new
// shape, so beta2 and the bias correction change the factors. Grafting removes
// a pure scale, so the caller also compares the preconditioned block.
static void shampoo_gradient_body(const float* G0, const float* H, float* Gs,
                                  int count, int step) {
  if (!H) {
    if (Gs != G0) {
#pragma omp for simd schedule(static)
      for (int i = 0; i < count; i++) Gs[i] = G0[i];
    }
    return;
  }
  float s = (float)(step - 1);
#pragma omp for simd schedule(static)
  for (int i = 0; i < count; i++) Gs[i] = G0[i] + s * H[i];
}

static inline void shampoo_gradient_at_step(const float* G0, const float* H,
                                            float* Gs, int count, int step) {
  // main() calls this outside the reference team. Inside ref_shampoo every
  // thread must enter the body so the worksharing loops stay aligned.
  if (omp_get_level() == 0) {
#pragma omp parallel num_threads(shampoo_threads(count))
    shampoo_gradient_body(G0, H, Gs, count, step);
  } else {
    shampoo_gradient_body(G0, H, Gs, count, step);
  }
}

// Absolute error within 0.1% of the peak. A rectangular factor is rank
// deficient, and the relative ridge amplifies the last bits of GEMM in a
// small subspace, so a few small entries can move by more than their own
// magnitude. This still rejects a wrong contraction or the wrong root.
static int shampoo_match(const char* name, const float* got, const float* refv,
                         int count) {
  double max_abs = 0.0, max_ref = 0.0;
  int finite = 1;
#pragma omp parallel for schedule(static) num_threads(shampoo_threads(count)) \
    reduction(max : max_abs, max_ref) reduction(& : finite)
  for (int i = 0; i < count; i++) {
    if (!std::isfinite(got[i]) || !std::isfinite(refv[i])) finite = 0;
    double d = fabs((double)got[i] - (double)refv[i]);
    double ar = fabs((double)refv[i]);
    if (d > max_abs) max_abs = d;
    if (ar > max_ref) max_ref = ar;
  }
  int ok = finite && max_ref > 0.0 && max_abs <= 1e-3 * max_ref + 1e-6;
  //printf("%s max abs error %g, peak %g\n", name, max_abs, max_ref);
  return ok;
}

// Fold one Gram sample into an EMA factor, bias-correct it, and replace the
// corrected copy by its inverse fourth root. `gram` is n x n.
static void fold_factor(float* F, float* Finv, const float* gram, int n,
                        float bc, float* work, double* partial) {
  const int count = n * n;
#pragma omp for simd schedule(static)
  for (int i = 0; i < count; i++) F[i] = kBeta2 * F[i] + kWeight * gram[i];
#pragma omp for simd schedule(static)
  for (int i = 0; i < count; i++) Finv[i] = F[i] / bc;
  ref_inverse_root(Finv, n, kShampooRoot, work, partial);
}

// `steps` preconditioner updates. H == nullptr repeats G0. Otherwise step s
// uses G0 + (s - 1) * H. L and R start at zero. Each step folds in the new
// Gram matrices,
//   L = beta2 L + (1 - beta2) G G^T
//   R = beta2 R + (1 - beta2) G^T G
// bias-corrects them, replaces each by its inverse fourth root, and writes
//   P = Linv^T G Rinv.
// That product shrinks G along the steep row combinations stored in L and
// the steep column combinations stored in R. The inverse root runs on every
// step, which is the amortized computation done with no reuse.
//
// AdaGrad grafting (Shi et al., arXiv:2309.06497, Algorithm 1) then replaces
// P by ||P_graft||_F / (||P||_F + 1e-16) * P, where
//   A = A + G⊙²,   P_graft = G / (sqrt(A) + 1e-10).
// A starts at zero. Each step adds G⊙², so A is the running sum of squared
// gradient entries. The ratio gives P the Frobenius norm of the grafted
// AdaGrad direction, which transfers that step length onto Shampoo.
// On success writes the final factors, the last preconditioned block, and the
// grafted direction. Returns 1 if a host allocation fails.
static int ref_shampoo(const float* G0, const float* H, float* L_out,
                       float* R_out, float* pre, float* out, int M, int N,
                       int steps) {
  const int lcount = M * M;
  const int rcount = N * N;
  const int gcount = M * N;
  const int nmax = (M > N) ? M : N;

  float* L = (float*)calloc(lcount, sizeof(float));
  float* R = (float*)calloc(rcount, sizeof(float));
  float* outer = (float*)malloc(sizeof(float) * (size_t)nmax * nmax);
  float* Linv = (float*)malloc(sizeof(float) * (size_t)lcount);
  float* Rinv = (float*)malloc(sizeof(float) * (size_t)rcount);
  float* tmp = (float*)malloc(sizeof(float) * (size_t)gcount);
  float* Gs = (float*)malloc(sizeof(float) * (size_t)gcount);
  float* work = (float*)malloc(sizeof(float) * (size_t)4 * nmax * nmax);
  float* A = (float*)calloc(gcount, sizeof(float));
  const int threads = shampoo_threads(M > N ? M : N);
  double* partial =
      (double*)calloc((size_t)threads * kPartialStride, sizeof(double));
  if (!L || !R || !outer || !Linv || !Rinv || !tmp || !Gs || !work || !A ||
      !partial) {
    fprintf(stderr, "reference allocation failed\n");
    free(L); free(R); free(outer); free(Linv); free(Rinv);
    free(tmp); free(Gs); free(work); free(A); free(partial);
    return 1;
  }

  // One team for every step. The helpers below are worksharing loops, so
  // each thread has to enter them; a second parallel region would nest.
#pragma omp parallel num_threads(threads)
  {
    for (int step = 1; step <= steps; step++) {
      const float* G = Gs;
      shampoo_gradient_at_step(G0, H, Gs, gcount, step);
      float bc = shampoo_bias_correction2(step);
      gram_left(G, outer, M, N);
      fold_factor(L, Linv, outer, M, bc, work, partial);
      gram_right(G, outer, M, N);
      fold_factor(R, Rinv, outer, N, bc, work, partial);

      gemm_tn(Linv, G, tmp, M, N, M);       // Linv^T @ G
      gemm_nn(tmp, Rinv, out, M, N, N);     // @ Rinv
#pragma omp single
      {
        if (step == steps) memcpy(pre, out, sizeof(float) * (size_t)gcount);
      }

#pragma omp for simd schedule(static)
      for (int i = 0; i < gcount; i++) A[i] += G[i] * G[i];
#pragma omp for simd schedule(static)
      for (int i = 0; i < gcount; i++)
        tmp[i] = G[i] / (sqrtf(A[i]) + kGraftEpsilon);
      float nrm_s = (float)sqrt(frobenius_sq(out, gcount, partial));
      float nrm_g = (float)sqrt(frobenius_sq(tmp, gcount, partial));
      float scale = nrm_g / (nrm_s + kGraftNormEps);
#pragma omp for simd schedule(static)
      for (int i = 0; i < gcount; i++) out[i] *= scale;
    }
  }

  memcpy(L_out, L, sizeof(float) * (size_t)lcount);
  memcpy(R_out, R, sizeof(float) * (size_t)rcount);
  free(L);
  free(R);
  free(outer);
  free(Linv);
  free(Rinv);
  free(tmp);
  free(Gs);
  free(work);
  free(A);
  free(partial);
  return 0;
}

#endif
