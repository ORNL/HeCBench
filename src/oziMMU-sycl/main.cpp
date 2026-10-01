// Benchmark derived from the "Accelerator for ozIMMU" project
// (https://github.com/RIKEN-RCCS/accelerator_for_ozIMMU), which enhances the
// Ozaki scheme implementation ozIMMU (https://github.com/enp1s0/ozIMMU).
//
// The Ozaki scheme emulates a double-precision GEMM (C = A * B) on integer
// matrix-multiplication units: each FP64 input matrix is split into a number of
// INT8 "slices", the slice pairs are multiplied with INT8 GEMMs producing INT32
// partial products, and the partial products are accumulated back into FP64.
//
// This benchmark reproduces the ozIMMU_H configuration of the repository
// (src_nearest_split + errfree_sum) plus the INT8-GEMM n-blocking (acc):
//   1. round-to-nearest splitting (split kernel);
//   2. group-wise error-free summation - slice-pair products that share the
//      same accumulation weight are summed directly in INT32 (oneMKL beta = 1),
//      so only one FP64 accumulation pass is needed per group;
//   3. n-blocking of the INT8 GEMM for large matrices.
//
// The INT8 and FP64 GEMMs use oneMKL; the split, accumulation and rescale run
// as SYCL kernels.  Accuracy is verified against the FP64 GEMM (and, for small
// sizes, a host reference); performance of the emulated DGEMM is reported.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <random>
#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include "../oziMMU-cuda/reference.h"

// Number of mantissa bits packed into each INT8 slice, following ozIMMU:
// min(7, floor((31 - ceil(log2(k))) / 2)).
static unsigned get_bits_per_int8(unsigned k) {
  if (k == 0) return 0;
  unsigned log2_k = 0;
  while ((1u << (log2_k + 1)) <= k) log2_k++;
  if ((1u << log2_k) != k) log2_k++;
  unsigned v = (31u - log2_k) / 2u;
  return v < 7u ? v : 7u;
}

// ceil(log2(k))
static unsigned ceil_log2(unsigned k) {
  unsigned l = 0;
  while ((1u << l) < k) l++;
  return l;
}

// Pad a leading dimension to a multiple of 4 (INT8 alignment).
static inline unsigned padded_ld(unsigned n) { return ((n + 3u) / 4u) * 4u; }

// ozIMMU's exponent helper: essentially ceil(log2(x)).
static inline short get_exp_npt(double x) {
  x *= 18014398509481984.0;  // 2^54
  uint64_t bits = sycl::bit_cast<uint64_t>(x);
  short exponent = (short)((bits >> 52) & 0x7FF) - 1077;
  return exponent + ((bits & 0xFFFFFFFFFFFFFULL) != 0);
}

constexpr int BLOCK = 256;

//============================================================================
// Round-to-nearest split of an FP64 matrix into INT8 slices (ozIMMU_RN).
// One work-group processes one logical row (a column of A, or a column of B).
//============================================================================
static void split_nearest(sycl::queue &q, int8_t *out_ptr, unsigned ldo,
                          double *sft, unsigned m, unsigned n,
                          const double *in_ptr, unsigned ld, unsigned num_split,
                          unsigned bits, bool col_major) {
  const size_t rows = m;
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<double, 1> lmem(sycl::range<1>(BLOCK), h);
    h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(rows * BLOCK), sycl::range<1>(BLOCK)),
        [=](sycl::nd_item<1> it) {
          const unsigned row_index = it.get_group(0);
          const unsigned lid = it.get_local_id(0);

          // Row maximum-magnitude reduction.
          double amax = 0.0;
          for (unsigned i = lid; i < n; i += BLOCK) {
            const double v = sycl::fabs(
                in_ptr[col_major ? ((size_t)i * ld + row_index)
                                 : ((size_t)i + (size_t)row_index * ld)]);
            amax = sycl::fmax(amax, v);
          }
          lmem[lid] = amax;
          it.barrier(sycl::access::fence_space::local_space);
          for (unsigned s = BLOCK / 2; s > 0; s >>= 1) {
            if (lid < s) lmem[lid] = sycl::fmax(lmem[lid], lmem[lid + s]);
            it.barrier(sycl::access::fence_space::local_space);
          }
          amax = lmem[0];

          const short log2_npt = get_exp_npt(amax);
          double t = sycl::ldexp(1.5, 53 - (int)bits + log2_npt);
          const double s_inc = (double)(1u << bits);
          double s = sycl::ldexp(s_inc, -1 - log2_npt);  // slice-0 scale
          const double t_inc = 1.0 / s_inc;
          const double base_scale = 1.0 / s;

          const size_t N = (size_t)m * ldo;
          unsigned i;
          for (i = lid; i < n; i += BLOCK) {
            double a =
                in_ptr[col_major ? ((size_t)i * ld + row_index)
                                 : ((size_t)i + (size_t)row_index * ld)];
            int8_t *dst = out_ptr + (size_t)row_index * ldo + i;
            double tt = t, ss = s;
            for (unsigned si = 0; si < num_split; si++) {
              // `volatile` prevents the compiler from reassociating the
              // round-to-nearest transform (a+tt)-tt into `a` under fast-math
              // (e.g. icpx -fp-model=fast), which would collapse the split.
              volatile double hi = a + tt;
              double a_i = hi - tt;
              a -= a_i;
              dst[N * si] = (int8_t)(a_i * ss);
              tt *= t_inc;
              ss *= s_inc;
            }
          }
          for (; i < ldo; i += BLOCK) {  // zero padding
            int8_t *dst = out_ptr + (size_t)row_index * ldo + i;
            for (unsigned si = 0; si < num_split; si++) dst[N * si] = 0;
          }

          if (lid == 0) sft[row_index] = base_scale;
        });
  });
}

//============================================================================
// Emulated DGEMM (C = A * B) via ozIMMU_H.
//============================================================================
static void ozimmu_gemm(sycl::queue &q, unsigned m, unsigned n, unsigned k,
                        unsigned num_split, const double *dA, const double *dB,
                        double *dC, int8_t *a_slices, int8_t *b_slices,
                        double *sft_a, double *sft_b, int32_t *c_i32,
                        const int32_t *co_zero) {
  const unsigned bits = get_bits_per_int8(k);
  const unsigned ld_int8_a = padded_ld(k);
  const unsigned ld_int8_b = padded_ld(k);
  const size_t slice_a_elems = (size_t)m * ld_int8_a;
  const size_t slice_b_elems = (size_t)n * ld_int8_b;
  const size_t mn = (size_t)m * n;

  // 1. Round-to-nearest split of A and B into INT8 slices.
  split_nearest(q, a_slices, ld_int8_a, sft_a, m, k, dA, m, num_split, bits,
                true);
  split_nearest(q, b_slices, ld_int8_b, sft_b, n, k, dB, k, num_split, bits,
                false);

  // Accumulate directly into the column-major output dC (ld = m): the
  // accumulator and dC share the same layout, so the group-wise error-free
  // sums are folded straight into dC and no final copy pass is needed.
  q.memset(dC, 0, mn * sizeof(double));

  int lim_bits = 31 - 2 * (int)bits - (int)ceil_log2(k);
  const unsigned group_cap = (lim_bits <= 0) ? 1u : (1u << lim_bits);

  const int32_t alpha_i = 1;
  const int64_t kc = ld_int8_a;

  // 2. Slice pairs grouped by sum = A_id + B_id share the accumulation weight
  //    2^(-bits*(sum-2)); sum them in INT32 and fold into FP64 once per group.
  for (unsigned sum = 2; sum <= num_split + 1; sum++) {
    const double scale = std::ldexp(1.0, -(int)bits * (int)(sum - 2));

    unsigned in_group = 0;
    for (unsigned j = 1; j < sum; j++) {
      const unsigned sa = j - 1;
      const unsigned sb = sum - 1 - j;
      if (sa >= num_split || sb >= num_split) continue;

      const int32_t beta_i = (in_group == 0) ? 0 : 1;
      const int8_t *Aptr = a_slices + sa * slice_a_elems;
      const int8_t *Bptr = b_slices + sb * slice_b_elems;

      // n-blocking optimization (acc).  gemm_bias computes
      // C = alpha * (op(A) - ao) * (op(B) - bo) + beta * C (+ co).
      // With ao = bo = 0 and co = 0 this is the plain signed INT8 GEMM
      // C = alpha * op(A) * op(B) + beta * C in INT32.
      const float alpha_f = (float)alpha_i;
      const float beta_f = (float)beta_i;
      size_t rem = n, offset = 0;
      while (rem > 0) {
        size_t nn = (rem <= 12288) ? rem : 8192;
        oneapi::mkl::blas::column_major::gemm_bias(
            q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans,
            oneapi::mkl::offset::fix, (int64_t)m, (int64_t)nn, kc, alpha_f, Aptr,
            (int64_t)ld_int8_a, (int8_t)0, Bptr + offset * ld_int8_b,
            (int64_t)ld_int8_b, (int8_t)0, beta_f, c_i32 + offset * m,
            (int64_t)m, co_zero);
        offset += nn;
        rem -= nn;
      }
      in_group++;

      const bool last_in_sum = (j == sum - 1);
      if (in_group == group_cap || last_in_sum) {
        const double *sa_ptr = sft_a;
        const double *sb_ptr = sft_b;
        const int32_t *ci = c_i32;
        double *cf = dC;
        const unsigned mm = m;
        q.parallel_for(sycl::range<1>(mn), [=](sycl::id<1> idx) {
          const size_t tid = idx[0];
          const unsigned mi = tid % mm;
          const unsigned ni = tid / mm;
          cf[tid] += (double)ci[tid] * sa_ptr[mi] * sb_ptr[ni] * scale;
        });
        in_group = 0;
      }
    }
  }
}

int main(int argc, char *argv[]) {
  if (argc != 4) {
    printf("Usage: %s <matrix size> <num split> <repeat>\n", argv[0]);
    printf("  emulated DGEMM C = A * B for square matrices of the given size\n");
    return 1;
  }
  const unsigned N = atoi(argv[1]);
  const unsigned num_split = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const unsigned m = N, n = N, k = N;
  const size_t mk = (size_t)m * k;
  const size_t kn = (size_t)k * n;
  const size_t mn = (size_t)m * n;

  printf("Matrix size: %u x %u x %u, INT8 slices: %u, bits/int8: %u\n", m, n, k,
         num_split, get_bits_per_int8(k));

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  double *A = (double *)malloc(mk * sizeof(double));
  double *B = (double *)malloc(kn * sizeof(double));
  double *C = (double *)malloc(mn * sizeof(double));
  double *C_ref = (double *)malloc(mn * sizeof(double));

  std::mt19937 gen(123);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (size_t i = 0; i < mk; i++) A[i] = dist(gen);
  for (size_t i = 0; i < kn; i++) B[i] = dist(gen);

  double *dA = sycl::malloc_device<double>(mk, q);
  double *dB = sycl::malloc_device<double>(kn, q);
  double *dC = sycl::malloc_device<double>(mn, q);
  q.memcpy(dA, A, mk * sizeof(double));
  q.memcpy(dB, B, kn * sizeof(double));

  const unsigned ld_int8 = padded_ld(k);
  int8_t *a_slices = sycl::malloc_device<int8_t>((size_t)num_split * m * ld_int8, q);
  int8_t *b_slices = sycl::malloc_device<int8_t>((size_t)num_split * n * ld_int8, q);
  double *sft_a = sycl::malloc_device<double>(m, q);
  double *sft_b = sycl::malloc_device<double>(n, q);
  int32_t *c_i32 = sycl::malloc_device<int32_t>(mn, q);
  int32_t *co_zero = sycl::malloc_device<int32_t>(1, q);  // zero C-offset for gemm_bias
  q.memset(co_zero, 0, sizeof(int32_t));
  q.wait();

  // Reference: oneMKL FP64 GEMM (true FP64).
  oneapi::mkl::blas::column_major::gemm(
      q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
      (int64_t)m, (int64_t)n, (int64_t)k, 1.0, dA, (int64_t)m, dB, (int64_t)k,
      0.0, dC, (int64_t)m);
  q.wait();
  q.memcpy(C_ref, dC, mn * sizeof(double)).wait();

  // Correctness of the reference against a host GEMM for small sizes.
  if (N <= 512) {
    double *C_host = (double *)malloc(mn * sizeof(double));
    reference(m, n, k, A, B, C_host);
    double max_rel = 0.0;
    for (size_t i = 0; i < mn; i++) {
      double d = fabs(C_host[i] - C_ref[i]);
      double r = d / (fabs(C_host[i]) + 1e-30);
      if (r > max_rel) max_rel = r;
    }
    printf("oneMKL DGEMM max relative error vs host: %e\n", max_rel);
    free(C_host);
  }

  // Emulated DGEMM via ozIMMU_H.
  ozimmu_gemm(q, m, n, k, num_split, dA, dB, dC, a_slices, b_slices, sft_a,
              sft_b, c_i32, co_zero);
  q.wait();
  q.memcpy(C, dC, mn * sizeof(double)).wait();

  double max_rel = 0.0, max_abs = 0.0;
  for (size_t i = 0; i < mn; i++) {
    double d = fabs(C[i] - C_ref[i]);
    if (d > max_abs) max_abs = d;
    double r = d / (fabs(C_ref[i]) + 1e-30);
    if (r > max_rel) max_rel = r;
  }
  printf("Emulated DGEMM vs oneMKL DGEMM: max abs error = %e, max rel error = %e\n",
         max_abs, max_rel);

  // Timing.
  q.wait();
  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
    ozimmu_gemm(q, m, n, k, num_split, dA, dB, dC, a_slices, b_slices, sft_a,
                sft_b, c_i32, co_zero);
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  double ms =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() *
      1e-6 / repeat;
  const double gflops = 2.0 * (double)m * n * k / (ms * 1e6);
  printf("Average execution time of emulated DGEMM: %f (ms), %.2f GFLOPS\n", ms,
         gflops);

  // Native oneMKL DGEMM performance (warmup + timed loop) for comparison.
  oneapi::mkl::blas::column_major::gemm(
      q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
      (int64_t)m, (int64_t)n, (int64_t)k, 1.0, dA, (int64_t)m, dB, (int64_t)k,
      0.0, dC, (int64_t)m);  // warmup
  q.wait();
  auto blas_start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
    oneapi::mkl::blas::column_major::gemm(
        q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
        (int64_t)m, (int64_t)n, (int64_t)k, 1.0, dA, (int64_t)m, dB, (int64_t)k,
        0.0, dC, (int64_t)m);
  }
  q.wait();
  auto blas_end = std::chrono::steady_clock::now();
  double blas_ms =
      std::chrono::duration_cast<std::chrono::nanoseconds>(blas_end - blas_start)
          .count() * 1e-6 / repeat;
  const double blas_gflops = 2.0 * (double)m * n * k / (blas_ms * 1e6);
  printf("Average execution time of oneMKL DGEMM: %f (ms), %.2f GFLOPS\n",
         blas_ms, blas_gflops);

  sycl::free(dA, q); sycl::free(dB, q); sycl::free(dC, q);
  sycl::free(a_slices, q); sycl::free(b_slices, q);
  sycl::free(sft_a, q); sycl::free(sft_b, q);
  sycl::free(c_i32, q); sycl::free(co_zero, q);
  free(A); free(B); free(C); free(C_ref);
  return 0;
}
