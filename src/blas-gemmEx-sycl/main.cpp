#include <stdio.h>
#include <chrono>
#include <cmath>
#include <sycl/sycl.hpp>
#ifdef USE_ONEMATH
#include <oneapi/math.hpp>
namespace math_ns = oneapi::math;
enum class compute_mode_t { standard, float_to_bf16, float_to_tf32 };
#else
#include <oneapi/mkl.hpp>
namespace math_ns = oneapi::mkl;
using compute_mode_t = math_ns::blas::compute_mode;
#endif
#include "utils.h"

#if defined(__SYCL_COMPILER_VERSION) && !defined(USE_ONEMATH)
using sycl::ext::oneapi::bfloat16;
#endif

template <typename T, typename S>
void allocate_memory(sycl::queue &q, int m, int n, int k, T **A, T **B, S **C) {
  *A = sycl::malloc_shared<T> (m * k, q);
  *B = sycl::malloc_shared<T> (k * n, q);
  *C = sycl::malloc_shared<S> (m * n, q);
}

template <typename T, typename S>
void free_memory(sycl::queue &q, T *A, T *B, S *C) {
  sycl::free(A, q);
  sycl::free(B, q);
  sycl::free(C, q);
}

void compute_reference(const double *A, const double *B, double *ref,
                        int rows, int n, int k)
{
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int p = 0; p < k; ++p)
        sum += A[i * k + p] * B[p * n + j];
      ref[i * n + j] = sum;
    }
  }
}

template <typename Tc>
bool verify_gemm(const char *name, const double *ref, const Tc *C,
                  int rows, int n, double tol, double scale = 1.0)
{
  double max_err = 0.0;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < n; ++j) {
      double val = double(C[i * n + j]) * scale;
      double err = fabs(val - ref[i * n + j]);
      if (err > max_err) max_err = err;
    }
  }
  bool ok = max_err <= tol;
  printf("%-32s %s (max abs error = %.6f, tol = %.6f)\n",
         name, ok ? "PASS" : "FAIL", max_err, tol);
  return ok;
}

template <typename Ta, typename Tc, typename Ts>
bool mkl_gemm_ex(
    sycl::queue &q,
    math_ns::transpose transA,
    math_ns::transpose transB,
    int m, int n, int k,
    Ta *A, Ta *B, Tc *C,
    int lda, int ldb, int ldc,
    const Ts alpha, const Ts beta,
    compute_mode_t mode =
      compute_mode_t::standard)
{
  sycl::event status;
  try {
#ifdef USE_ONEMATH
    // no compute mode equivalent in oneMath yet
    (void)mode;
    status = math_ns::blas::column_major::gemm(
      q,
      transA, transB,
      m, n, k,
      alpha, A, lda,
      B, ldb, beta,
      C, ldc);
#else
    status = math_ns::blas::column_major::gemm(
      q,
      transA, transB,
      m, n, k,
      alpha, A, lda,
      B, ldb, beta,
      C, ldc, mode);
#endif
  } catch(sycl::exception const& e) {
    std::cout << "\t\tCaught synchronous SYCL exception during GEMM:\n"
              << e.what() << std::endl;
    return false;
  }
  status.wait();
  return true;
}

template <typename Ta, typename Tc, typename Ts>
void test_gemm(sycl::queue &q,
               const int m, const int n, const int k,
               Ta *A, Ta *B, Tc *C,
               const Ts alpha, const Ts beta, int iteration,
               compute_mode_t mode =
                 compute_mode_t::standard)
{
  double total_time = 0;

  for (int i = 0; i < iteration; ++i) {
    auto start = std::chrono::steady_clock::now();
    bool success = mkl_gemm_ex(q,
                               math_ns::transpose::nontrans,
                               math_ns::transpose::nontrans,
                               n, // number of rows of matrix A and C
                               m, // number of columns of matrix B and C
                               k, // number of columns of A and rows of B
                               B,
                               A,
                               C,
                               n, // lda
                               k, // ldb
                               n, // ldc
                               alpha,
                               beta,
                               mode);
    auto end = std::chrono::steady_clock::now();
    if (!success) break;
    else if (i > 0) {
      total_time += std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    }
  }
  if (total_time > 0.0) {
    double avg_time = total_time / (iteration - 1);
    printf("%.3f ms\n", avg_time);
    performance(m, n, k, std::is_same<Ta, int8_t>::value, avg_time * 1e-3);
  }
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <M> <N> <K> <iterations>\n", argv[0]);
    printf("C = A X B (A: M * K, B: K * N, C: M * N)\n");
    return 1;
  }
  const int m = atoi(argv[1]);
  const int n = atoi(argv[2]);
  const int k = atoi(argv[3]);
  const int iteration = atoi(argv[4]);

  printf("shape: (%d, %d) x (%d, %d)\n", m, k, k, n);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  const double d_alpha = 1.0, d_beta = 0.0;
  const float f_alpha = 1.f, f_beta = 0.f;
  sycl::half h_alpha = sycl::vec<float, 1>{1.f}
                           .convert<sycl::half, sycl::rounding_mode::rte>()[0],
             h_beta = sycl::vec<float, 1>{0.f}
                          .convert<sycl::half, sycl::rounding_mode::rte>()[0];
#if defined(__SYCL_COMPILER_VERSION) && !defined(USE_ONEMATH)
  const float bf_alpha = 1.f, bf_beta = 0.f;
  const int32_t i_alpha = 1, i_beta = 0;
#endif

  double *dA, *dB, *dC;
  float *fA, *fB, *fC;
  sycl::half *hA, *hB, *hC;
#if defined(__SYCL_COMPILER_VERSION) && !defined(USE_ONEMATH)
  bfloat16 *bfA, *bfB, *bfC;
  int8_t *iA, *iB; int32_t *iC;
#endif

  allocate_memory(q, m, n, k, &dA, &dB, &dC);
  allocate_memory(q, m, n, k, &fA, &fB, &fC);
  allocate_memory(q, m, n, k, &hA, &hB, &hC);
#if defined(__SYCL_COMPILER_VERSION) && !defined(USE_ONEMATH)
  allocate_memory(q, m, n, k, &bfA, &bfB, &bfC);
  allocate_memory(q, m, n, k, &iA, &iB, &iC);
#endif

  for (int i = 0; i < m * k; ++i) {
    dA[i] = double(i % 255 - 127) / 127;
    fA[i] = float(i % 255 - 127) / 127;
    hA[i] = sycl::vec<float, 1>{fA[i]}
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];
#if defined(__SYCL_COMPILER_VERSION) && !defined(USE_ONEMATH)
    bfA[i] = bfloat16(fA[i]);
    iA[i] = float2int8(fA[i], 127);
#endif
  }
  for (int i = 0; i < k * n; ++i) {
    dB[i] = double(i % 255 - 127) / 127;
    fB[i] = float(i % 255 - 127) / 127;
    hB[i] = sycl::vec<float, 1>{fB[i]}
                .convert<sycl::half, sycl::rounding_mode::rte>()[0];
#if defined(__SYCL_COMPILER_VERSION) && !defined(USE_ONEMATH)
    bfB[i] = bfloat16(fB[i]);
    iB[i] = float2int8(fB[i], 127);
#endif
  }

  const int verify_rows = (m < 8) ? m : 8;
  double *ref = (double*) malloc(verify_rows * n * sizeof(double));
  if (ref == nullptr) {
    printf("reference buffer allocation failed\n");
    return 1;
  }
  compute_reference(dA, dB, ref, verify_rows, n, k);
  const double base_tol = double(k);


  printf(">>>>>>>>>>>>>>>>> test fp64 >>>>>>>>>>>>>>>>>\n");
  test_gemm(q, m, n, k, dA, dB, dC, d_alpha, d_beta, iteration);
  verify_gemm("fp64", ref, dC, verify_rows, n, 1e-6 * base_tol);

#if defined(__SYCL_COMPILER_VERSION) && !defined(USE_ONEMATH)
  printf(">>>>>>>>>>>>>>>>> test fp32 (compute mode bf16) >>>>>>>>>>>>>>>>>\n");
  test_gemm(q, m, n, k, fA, fB, fC, f_alpha, f_beta, iteration,
            compute_mode_t::float_to_bf16);
  verify_gemm("fp32 (bf16 compute mode)", ref, fC, verify_rows, n, 5e-2 * base_tol);

  printf(">>>>>>>>>>>>>>>>> test fp32 (compute mode tf32) >>>>>>>>>>>>>>>>>\n");
  test_gemm(q, m, n, k, fA, fB, fC, f_alpha, f_beta, iteration,
            compute_mode_t::float_to_tf32);
  verify_gemm("fp32 (tf32 compute mode)", ref, fC, verify_rows, n, 1e-2 * base_tol);
#endif

  printf(">>>>>>>>>>>>>>>>> test fp32 (compute mode fp32) >>>>>>>>>>>>>>>>>\n");
  test_gemm(q, m, n, k, fA, fB, fC, f_alpha, f_beta, iteration);
  verify_gemm("fp32", ref, fC, verify_rows, n, 1e-3 * base_tol);

  printf(">>>>>>>>>>>>>>>>> test fp16 >>>>>>>>>>>>>>>>>\n");
  test_gemm(q, m, n, k, hA, hB, hC, h_alpha, h_beta, iteration);
  verify_gemm("fp16", ref, hC, verify_rows, n, 5e-2 * base_tol);

#if defined(__SYCL_COMPILER_VERSION) && !defined(USE_ONEMATH)
  printf(">>>>>>>>>>>>>>>>> test bfloat16 >>>>>>>>>>>>>\n");
  test_gemm(q, m, n, k, bfA, bfB, bfC, bf_alpha, bf_beta, iteration);
  verify_gemm("bf16", ref, bfC, verify_rows, n, 8e-2 * base_tol);

  printf(">>>>>>>>>>>>>>>>> test int8 >>>>>>>>>>>>>>>>>\n");
  test_gemm(q, m, n, k, iA, iB, iC, i_alpha, i_beta, iteration);
  verify_gemm("int8", ref, iC, verify_rows, n, 5e-1 * base_tol, 1.0 / (127.0 * 127.0));
#endif

  free(ref);


  printf(">>>>>>>>>>>>>>>>> compare result >>>>>>>>>>>>>>>>>\n");
  printf("fp64: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5lf%c", dC[i], " \n"[i==9]);

  printf("fp32: "); // fp32 (bf16 and tf32) are not compared
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", fC[i], " \n"[i==9]);

  printf("fp16: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", float(hC[i]), " \n"[i==9]);

#if defined(__SYCL_COMPILER_VERSION) && !defined(USE_ONEMATH)
  printf("bf16: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", float(bfC[i]), " \n"[i==9]);

  printf("int8: ");
  for (int i = 0; i < 10; ++i)
    printf("%.5f%c", float(iC[i])/127/127, " \n"[i==9]);
#endif

  free_memory(q, dA, dB, dC);
  free_memory(q, fA, fB, fC);
  free_memory(q, hA, hB, hC);
#if defined(__SYCL_COMPILER_VERSION) && !defined(USE_ONEMATH)
  free_memory(q, bfA, bfB, bfC);
  free_memory(q, iA, iB, iC);
#endif
  return 0;
}
