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
// (src_nearest_split + errfree_sum), which combines the two techniques the
// project introduces, together with the INT8-GEMM n-blocking (acc):
//   1. round-to-nearest splitting (split_nearest_kernel);
//   2. group-wise error-free summation - slice-pair products that share the
//      same accumulation weight are summed directly in INT32 (hipBLAS beta = 1),
//      so only one FP64 accumulation pass is needed per group;
//   3. n-blocking of the INT8 GEMM for large matrices.
//
// Accuracy is verified against hipBLAS DGEMM (and, for small sizes, a host
// reference); performance of the emulated DGEMM is reported.

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <chrono>
#include <random>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>
#include "../oziMMU-cuda/reference.h"

#define HIP_CHECK(call)                                                         \
  do {                                                                          \
    hipError_t err = (call);                                                    \
    if (err != hipSuccess) {                                                    \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(err),        \
              __FILE__, __LINE__);                                              \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  } while (0)

#define HIPBLAS_CHECK(call)                                                     \
  do {                                                                          \
    hipblasStatus_t st = (call);                                                \
    if (st != HIPBLAS_STATUS_SUCCESS) {                                         \
      fprintf(stderr, "hipBLAS error %d at %s:%d\n", (int)st, __FILE__,         \
              __LINE__);                                                        \
      exit(EXIT_FAILURE);                                                       \
    }                                                                           \
  } while (0)

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
__device__ short get_exp_npt(double x) {
  x *= 18014398509481984.0;  // 2^54
  uint64_t bits = __double_as_longlong(x);
  short exponent = (short)((bits >> 52) & 0x7FF) - 1077;
  return exponent + ((bits & 0xFFFFFFFFFFFFFULL) != 0);
}

// Reduce the maximum absolute value across one logical row.
__device__ double block_amax(const double *ptr, unsigned length, unsigned inc,
                             double *shm) {
  double amax = 0.0;
  for (unsigned i = threadIdx.x; i < length; i += blockDim.x) {
    double t = fabs(ptr[(size_t)i * inc]);
    if (t > amax) amax = t;
  }
  for (unsigned off = 16; off >= 1; off >>= 1) {
    double t = __shfl_xor(amax, off);
    if (t > amax) amax = t;
  }
  if ((threadIdx.x & 0x1f) == 0) shm[threadIdx.x >> 5] = amax;
  __syncthreads();
  if (threadIdx.x < 32) {
    amax = (threadIdx.x < (blockDim.x >> 5)) ? shm[threadIdx.x] : 0.0;
    for (unsigned off = 16; off >= 1; off >>= 1) {
      double t = __shfl_xor(amax, off);
      if (t > amax) amax = t;
    }
    if (threadIdx.x == 0) shm[0] = amax;
  }
  __syncthreads();
  return shm[0];
}

//============================================================================
// Round-to-nearest split of an FP64 matrix into INT8 slices (ozIMMU_RN).
//============================================================================
__global__ void split_nearest_kernel(int8_t *__restrict__ out_ptr,
                                      const unsigned ldo,
                                      double *__restrict__ sft, const unsigned m,
                                      const unsigned n,
                                      const double *__restrict__ in_ptr,
                                      const unsigned ld,
                                      const unsigned num_split,
                                      const unsigned bits, const bool col_major) {
  __shared__ double smem[8];
  const unsigned row_index = blockIdx.x;

  const double amax = block_amax(
      in_ptr + (col_major ? row_index : (size_t)row_index * ld), n,
      col_major ? ld : 1, smem);

  const short log2_npt = get_exp_npt(amax);
  double t = scalbn(1.5, 53 - (int)bits + log2_npt);
  const double s_inc = (double)(1u << bits);
  double s = scalbn(s_inc, -1 - log2_npt);  // slice-0 scale s_init
  const double t_inc = 1.0 / s_inc;
  const double base_scale = 1.0 / s;        // weight of slice 0 (= 1 / s_init)

  const size_t N = (size_t)m * ldo;
  unsigned i;
  for (i = threadIdx.x; i < n; i += blockDim.x) {
    double a =
        in_ptr[col_major ? ((size_t)i * ld + row_index)
                         : ((size_t)i + (size_t)row_index * ld)];
    int8_t *dst = out_ptr + (size_t)row_index * ldo + i;
    double tt = t, ss = s;
    for (unsigned si = 0; si < num_split; si++) {
      // `volatile` keeps the round-to-nearest transform (a+tt)-tt from being
      // reassociated into `a` under fast-math, which would collapse the split.
      volatile double hi = a + tt;  // round to current slice granularity
      double a_i = hi - tt;
      a -= a_i;                     // residual for the next slice
      dst[N * si] = (int8_t)(a_i * ss);
      tt *= t_inc;
      ss *= s_inc;
    }
  }
  for (; i < ldo; i += blockDim.x) {  // zero padding
    int8_t *dst = out_ptr + (size_t)row_index * ldo + i;
    for (unsigned si = 0; si < num_split; si++) dst[N * si] = 0;
  }

  if (threadIdx.x == 0) sft[row_index] = base_scale;
}

// Error-free-summation accumulate: the INT32 result already holds the sum of
// one slice-pair group, all sharing the same accumulation weight `scale`.
__global__ void accumulate_ef_kernel(const unsigned m,
                                     double *__restrict__ f64_ptr,
                                     const int32_t *__restrict__ i32_ptr,
                                     const size_t length,
                                     const double *__restrict__ sft_a,
                                     const double *__restrict__ sft_b,
                                     const double scale) {
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= length) return;
  const unsigned mi = tid % m;
  const unsigned ni = tid / m;
  f64_ptr[tid] += (double)i32_ptr[tid] * sft_a[mi] * sft_b[ni] * scale;
}

__global__ void init_buffer_kernel(double *__restrict__ ptr,
                                    const size_t length) {
  const size_t tid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < length) ptr[tid] = 0.0;
}

//============================================================================
// Emulated DGEMM (C = A * B) via ozIMMU_H.
//============================================================================
static void ozimmu_gemm(hipblasHandle_t blas, unsigned m, unsigned n,
                        unsigned k, unsigned num_split, const double *dA,
                        const double *dB, double *dC, int8_t *a_slices,
                        int8_t *b_slices, double *sft_a, double *sft_b,
                        int32_t *c_i32) {
  const unsigned bits = get_bits_per_int8(k);
  const unsigned ld_int8_a = padded_ld(k);
  const unsigned ld_int8_b = padded_ld(k);
  const size_t slice_a_elems = (size_t)m * ld_int8_a;
  const size_t slice_b_elems = (size_t)n * ld_int8_b;
  const size_t mn = (size_t)m * n;

  // 1. Round-to-nearest split of A and B into INT8 slices.
  split_nearest_kernel<<<m, 256>>>(a_slices, ld_int8_a, sft_a, m, k, dA, m,
                                   num_split, bits, true);
  split_nearest_kernel<<<n, 256>>>(b_slices, ld_int8_b, sft_b, n, k, dB, k,
                                   num_split, bits, false);

  // Accumulate directly into the column-major output dC (ld = m): the
  // accumulator and dC share the same layout, so the group-wise error-free
  // sums are folded straight into dC and no final copy pass is needed.
  init_buffer_kernel<<<(mn + 255) / 256, 256>>>(dC, mn);

  int lim_bits = 31 - 2 * (int)bits - (int)ceil_log2(k);
  const unsigned group_cap = (lim_bits <= 0) ? 1u : (1u << lim_bits);

  const int alpha_i = 1;
  const int kc = (int)ld_int8_a;

  // 2. Slice pairs grouped by sum = A_id + B_id share the accumulation weight
  //    2^(-bits*(sum-2)); sum them in INT32 and fold into FP64 once per group.
  for (unsigned sum = 2; sum <= num_split + 1; sum++) {
    const double scale = ldexp(1.0, -(int)bits * (int)(sum - 2));

    unsigned in_group = 0;
    for (unsigned j = 1; j < sum; j++) {
      const unsigned sa = j - 1;
      const unsigned sb = sum - 1 - j;
      if (sa >= num_split || sb >= num_split) continue;

      const int beta_i = (in_group == 0) ? 0 : 1;
      const int8_t *Aptr = a_slices + sa * slice_a_elems;
      const int8_t *Bptr = b_slices + sb * slice_b_elems;

      // n-blocking optimization (acc).
      size_t rem = n, offset = 0;
      while (rem > 0) {
        size_t nn = (rem <= 12288) ? rem : 8192;
        HIPBLAS_CHECK(hipblasGemmEx(
            blas, HIPBLAS_OP_T, HIPBLAS_OP_N, (int)m, (int)nn, kc, &alpha_i,
            Aptr, HIPBLAS_R_8I, (int)ld_int8_a, Bptr + offset * ld_int8_b,
            HIPBLAS_R_8I, (int)ld_int8_b, &beta_i, c_i32 + offset * m,
            HIPBLAS_R_32I, (int)m, HIPBLAS_COMPUTE_32I, HIPBLAS_GEMM_DEFAULT));
        offset += nn;
        rem -= nn;
      }
      in_group++;

      const bool last_in_sum = (j == sum - 1);
      if (in_group == group_cap || last_in_sum) {
        accumulate_ef_kernel<<<(mn + 255) / 256, 256>>>(m, dC, c_i32, mn, sft_a,
                                                        sft_b, scale);
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

  double *A = (double *)malloc(mk * sizeof(double));
  double *B = (double *)malloc(kn * sizeof(double));
  double *C = (double *)malloc(mn * sizeof(double));
  double *C_ref = (double *)malloc(mn * sizeof(double));

  std::mt19937 gen(123);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (size_t i = 0; i < mk; i++) A[i] = dist(gen);
  for (size_t i = 0; i < kn; i++) B[i] = dist(gen);

  double *dA, *dB, *dC;
  HIP_CHECK(hipMalloc(&dA, mk * sizeof(double)));
  HIP_CHECK(hipMalloc(&dB, kn * sizeof(double)));
  HIP_CHECK(hipMalloc(&dC, mn * sizeof(double)));
  HIP_CHECK(hipMemcpy(dA, A, mk * sizeof(double), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB, B, kn * sizeof(double), hipMemcpyHostToDevice));

  const unsigned ld_int8 = padded_ld(k);
  int8_t *a_slices, *b_slices;
  double *sft_a, *sft_b;
  int32_t *c_i32;
  HIP_CHECK(hipMalloc(&a_slices, (size_t)num_split * m * ld_int8));
  HIP_CHECK(hipMalloc(&b_slices, (size_t)num_split * n * ld_int8));
  HIP_CHECK(hipMalloc(&sft_a, m * sizeof(double)));
  HIP_CHECK(hipMalloc(&sft_b, n * sizeof(double)));
  HIP_CHECK(hipMalloc(&c_i32, mn * sizeof(int32_t)));

  hipblasHandle_t blas;
  HIPBLAS_CHECK(hipblasCreate(&blas));

  // Reference: hipBLAS DGEMM (true FP64).
  const double alpha = 1.0, beta = 0.0;
  HIPBLAS_CHECK(hipblasDgemm(blas, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, &alpha,
                             dA, m, dB, k, &beta, dC, m));
  HIP_CHECK(hipMemcpy(C_ref, dC, mn * sizeof(double), hipMemcpyDeviceToHost));

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
    printf("hipBLAS DGEMM max relative error vs host: %e\n", max_rel);
    free(C_host);
  }

  // Emulated DGEMM via ozIMMU_H.
  ozimmu_gemm(blas, m, n, k, num_split, dA, dB, dC, a_slices, b_slices, sft_a,
              sft_b, c_i32);
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(C, dC, mn * sizeof(double), hipMemcpyDeviceToHost));

  double max_rel = 0.0, max_abs = 0.0;
  for (size_t i = 0; i < mn; i++) {
    double d = fabs(C[i] - C_ref[i]);
    if (d > max_abs) max_abs = d;
    double r = d / (fabs(C_ref[i]) + 1e-30);
    if (r > max_rel) max_rel = r;
  }
  printf("Emulated DGEMM vs hipBLAS DGEMM: max abs error = %e, max rel error = %e\n",
         max_abs, max_rel);

  // Timing.
  HIP_CHECK(hipDeviceSynchronize());
  auto start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
    ozimmu_gemm(blas, m, n, k, num_split, dA, dB, dC, a_slices, b_slices, sft_a,
                sft_b, c_i32);
  }
  HIP_CHECK(hipDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  double ms =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() *
      1e-6 / repeat;
  const double gflops = 2.0 * (double)m * n * k / (ms * 1e6);
  printf("Average execution time of emulated DGEMM: %f (ms), %.2f GFLOPS\n", ms,
         gflops);

  // Native hipBLAS DGEMM performance (warmup + timed loop) for comparison.
  HIPBLAS_CHECK(hipblasDgemm(blas, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, &alpha,
                             dA, m, dB, k, &beta, dC, m));  // warmup
  HIP_CHECK(hipDeviceSynchronize());
  auto blas_start = std::chrono::steady_clock::now();
  for (int r = 0; r < repeat; r++) {
    HIPBLAS_CHECK(hipblasDgemm(blas, HIPBLAS_OP_N, HIPBLAS_OP_N, m, n, k, &alpha,
                               dA, m, dB, k, &beta, dC, m));
  }
  HIP_CHECK(hipDeviceSynchronize());
  auto blas_end = std::chrono::steady_clock::now();
  double blas_ms =
      std::chrono::duration_cast<std::chrono::nanoseconds>(blas_end - blas_start)
          .count() *
      1e-6 / repeat;
  const double blas_gflops = 2.0 * (double)m * n * k / (blas_ms * 1e6);
  printf("Average execution time of hipBLAS DGEMM: %f (ms), %.2f GFLOPS\n",
         blas_ms, blas_gflops);

  HIPBLAS_CHECK(hipblasDestroy(blas));
  HIP_CHECK(hipFree(dA)); HIP_CHECK(hipFree(dB)); HIP_CHECK(hipFree(dC));
  HIP_CHECK(hipFree(a_slices)); HIP_CHECK(hipFree(b_slices));
  HIP_CHECK(hipFree(sft_a)); HIP_CHECK(hipFree(sft_b));
  HIP_CHECK(hipFree(c_i32));
  free(A); free(B); free(C); free(C_ref);
  return 0;
}
