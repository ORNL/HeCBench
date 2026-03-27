#include <chrono>
#include <cmath>
#include <iostream>
#include <limits>  // std::numeric_limits
#include <string>
#include <vector>
#include <sycl/sycl.hpp>

typedef sycl::half fp16;
typedef float fp32;

#include "reference.h"

using namespace sycl::ext::oneapi::experimental::matrix;

// Matrix data initialization
template <typename DataT>
static inline void fill(DataT *mat, uint32_t m, uint32_t n) {
  srand(m * n);
  auto ld = n;
  for (uint32_t i = 0; i < m; ++i) {
    for (uint32_t j = 0; j < n; ++j) {
      // Ascending order for each neighboring element.
      // Alternate sign for even / odd
      auto value = (i * n + j) % 13;
      mat[i * ld + j] =
          (value % 3) ? -static_cast<DataT>(value) : static_cast<DataT>(value);
    }
  }
}

// : 16 x 16
const int WMMA_M = 16;
const int WMMA_N = 16;

// multiples of 16
const int WMMA_K = 16;

// Tile size
const int TILE_M = 64;
const int TILE_N = 64;

// This kernel assumes that each thread block has warpSize threads
// D = alpha * (A x B) + beta * C
//
// In this example, we assume:
// : A is in row-major format     (M x K)
// : B is in col-major format     (K x N)
// : C, D are in row-major format (M x N)
// : Multiplication is NOT in-place, output is written to D matrix
// : No LDS required
void gemm_impl0(const uint32_t m, const uint32_t n, const uint32_t k,
                fp16 const *__restrict__ a,
                fp16 const *__restrict__ b,
                fp32 const *c,
                fp32 *d, const uint32_t lda, const uint32_t ldb,
                const uint32_t ldc, const uint32_t ldd,
                const fp32 alpha, const fp32 beta,
                sycl::nd_item<3> &item)
{
  sycl::sub_group sg = item.get_sub_group();

  auto p_a = sycl::address_space_cast<sycl::access::address_space::global_space,
                                      sycl::access::decorated::no>(a);
  auto p_b = sycl::address_space_cast<sycl::access::address_space::global_space,
                                      sycl::access::decorated::no>(b);
  auto p_c = sycl::address_space_cast<sycl::access::address_space::global_space,
                                      sycl::access::decorated::no>(c);
  auto p_d = sycl::address_space_cast<sycl::access::address_space::global_space,
                                      sycl::access::decorated::no>(d);

  joint_matrix<sycl::sub_group, fp16, use::a, WMMA_M, WMMA_K, layout::row_major> sub_a;
  joint_matrix<sycl::sub_group, fp16, use::b, WMMA_K, WMMA_N, layout::col_major> sub_b;
  joint_matrix<sycl::sub_group, fp32, use::accumulator, WMMA_M, WMMA_N> sub_c;
  joint_matrix<sycl::sub_group, fp32, use::accumulator, WMMA_M, WMMA_N> sub_acc;

  joint_matrix_fill(sg, sub_acc, 0.0f);

  auto cRow = item.get_group(2) * WMMA_M;
  auto cCol = item.get_group(1) * WMMA_N;

  for (int i = 0; i < k; i += WMMA_K) {
    // Load the inputs
    joint_matrix_load(sg, sub_a, p_a + cRow * lda + i, lda);
    joint_matrix_load(sg, sub_b, p_b + cCol * ldb + i, ldb);
    joint_matrix_mad(sg, sub_acc, sub_a, sub_b, sub_acc);
  }

  joint_matrix_load(sg, sub_c, p_c + cRow * ldc + cCol, ldc, layout::row_major);

  joint_matrix_apply(sg, sub_acc, sub_c, [=] (const fp32 &acc, fp32 &c)
                                             {c = alpha * acc + beta * c;});

  joint_matrix_store(sg, sub_c, p_d + cRow * ldd + cCol, ldd, layout::row_major);
}

// D = alpha * (A x B) + beta * C
//
// In this example, we assume:
// : A is in row-major format     (M x K)
// : B is in col-major format     (K x N)
// : C, D are in row-major format (M x N)
// : Multiplication is NOT in-place, output is written to D matrix
// : No LDS required
//
template <unsigned int WAVE_SIZE>
void gemm_impl1(const uint32_t m, const uint32_t n, const uint32_t k,
                fp16 const *__restrict__ a,
                fp16 const *__restrict__ b,
                fp32 const *c,
                fp32 *d, const uint32_t lda, const uint32_t ldb,
                const uint32_t ldc, const uint32_t ldd,
                const fp32 alpha, const fp32 beta,
                sycl::nd_item<3> &item)
{
  sycl::sub_group sg = item.get_sub_group();

  auto p_a = sycl::address_space_cast<sycl::access::address_space::global_space,
                                      sycl::access::decorated::no>(a);
  auto p_b = sycl::address_space_cast<sycl::access::address_space::global_space,
                                      sycl::access::decorated::no>(b);
  auto p_c = sycl::address_space_cast<sycl::access::address_space::global_space,
                                      sycl::access::decorated::no>(c);
  auto p_d = sycl::address_space_cast<sycl::access::address_space::global_space,
                                      sycl::access::decorated::no>(d);

  joint_matrix<sycl::sub_group, fp16, use::a, WMMA_M, WMMA_K, layout::row_major> sub_a;
  joint_matrix<sycl::sub_group, fp16, use::b, WMMA_K, WMMA_N, layout::col_major> sub_b;
  joint_matrix<sycl::sub_group, fp32, use::accumulator, WMMA_M, WMMA_N> sub_c;
  joint_matrix<sycl::sub_group, fp32, use::accumulator, WMMA_M, WMMA_N> sub_acc;

  joint_matrix_fill(sg, sub_acc, 0.0f);

  // Map threadIdx to warpIdx
  auto warpIdx = item.get_local_id(2) / WAVE_SIZE;
  auto warpIdy = item.get_local_id(1);

  // Target C block
  auto cRow = item.get_group(2) * TILE_M + warpIdx * WMMA_M;
  auto cCol = item.get_group(1) * TILE_N + warpIdy * WMMA_N;

  // Bounds check
  for (int i = 0; i < k; i += WMMA_K) {
    // Load the inputs
    joint_matrix_load(sg, sub_a, p_a + cRow * lda + i, lda);
    joint_matrix_load(sg, sub_b, p_b + cCol * ldb + i, ldb);
    joint_matrix_mad(sg, sub_acc, sub_a, sub_b, sub_acc);
  }

  joint_matrix_load(sg, sub_c, p_c + cRow * ldc + cCol, ldc, layout::row_major);

  joint_matrix_apply(sg, sub_acc, sub_c, [=] (const fp32 &acc, fp32 &c)
                                             {c = alpha * acc + beta * c;});

  joint_matrix_store(sg, sub_c, p_d + cRow * ldd + cCol, ldd, layout::row_major);
}

void gemm_wmma(int impl, uint32_t m, uint32_t n, uint32_t k, fp32 alpha,
               fp32 beta, int32_t repeat, int32_t verify)
{
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  auto sg_sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
  auto r = std::max_element(sg_sizes.begin(), sg_sizes.end());
  int WAVE_SIZE = *r;

  // Bounds check
  if (impl == 0) {
    if (m < WMMA_M || n < WMMA_N || k < WMMA_K || m % WMMA_M || n % WMMA_N || k % WMMA_K) {
      std::cout << "Unsupported size!\n";
      return;
    }

  } else {
    if ((m < TILE_M) || n < TILE_N || k < WMMA_K || m % WMMA_M || n % WMMA_N || k % WMMA_K ||
        TILE_M / WMMA_M * WAVE_SIZE * TILE_N / WMMA_N > 1024) {
      std::cout << "Unsupported size!\n";
      return;
    }
  }

  int lda = k;
  int ldb = k;
  int ldc = n;
  int ldd = ldc;

  std::cout << "Initializing host data..." << std::endl;

  // Initialize input matrices
  std::vector<fp16> matrixA(m * k);
  std::vector<fp16> matrixB(k * n);
  std::vector<fp32> matrixC(m * n);
  // Fill outputs with NaN to catch contamination
  std::vector<fp32> matrixD(m * n, std::numeric_limits<fp32>::signaling_NaN());

  fill(matrixA.data(), m, k);
  fill(matrixB.data(), k, n);
  fill(matrixC.data(), m, n);

  std::cout << "Initializing device data..." << std::endl;

  const size_t bytesA = matrixA.size() * sizeof(fp16);
  const size_t bytesB = matrixB.size() * sizeof(fp16);
  const size_t bytesC = matrixC.size() * sizeof(fp32);
  const size_t bytesD = matrixD.size() * sizeof(fp32);

  fp16* d_a = sycl::malloc_device<fp16>(matrixA.size(), q);
  q.memcpy(d_a, matrixA.data(), bytesA);

  fp16* d_b = sycl::malloc_device<fp16>(matrixB.size(), q);
  q.memcpy(d_b, matrixB.data(), bytesB);

  fp32* d_c = sycl::malloc_device<fp32>(matrixC.size(), q);
  q.memcpy(d_c, matrixC.data(), bytesC);

  fp32* d_d = sycl::malloc_device<fp32>(matrixD.size(), q);
  q.memcpy(d_d, matrixD.data(), bytesD);

  std::cout << "Launching GEMM kernel..." << std::endl;

  // dim0, dim1, dim2
  sycl::range<3> gws {1, 1, 1};
  sycl::range<3> lws {1, 1, 1};

  if (impl == 0) {
    lws[2] = WAVE_SIZE;
    gws[2] = m / WMMA_M * lws[2];
    gws[1] = n / WMMA_N;
  }
  else {
    lws[2] = TILE_M / WMMA_M * WAVE_SIZE;
    lws[1] = TILE_N / WMMA_N;
    gws[2] = m / TILE_M * lws[2];
    gws[1] = n / TILE_N * lws[1];
  }

  auto gemm0Fn = [=](sycl::nd_item<3> item) {
    gemm_impl0(m, n, k, d_a, d_b, d_c, d_d, lda, ldb, ldc, ldd, alpha, beta, item);
  };
  auto gemm1w32Fn = [=](sycl::nd_item<3> item) {
    gemm_impl1<32>(m, n, k, d_a, d_b, d_c, d_d, lda, ldb, ldc, ldd, alpha, beta, item);
  };
  auto gemm1w64Fn = [=](sycl::nd_item<3> item) {
    gemm_impl1<64>(m, n, k, d_a, d_b, d_c, d_d, lda, ldb, ldc, ldd, alpha, beta, item);
  };

  for (uint32_t w = 0; w < 30; w++) {
    if (impl == 0)
      q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<3>(gws, lws), gemm0Fn);
      });
    else if (impl == 1) {
      if (WAVE_SIZE == 32)
        q.submit([&](sycl::handler &cgh) {
          cgh.parallel_for(sycl::nd_range<3>(gws, lws), gemm1w32Fn);
        });
      else
        q.submit([&](sycl::handler &cgh) {
          cgh.parallel_for(sycl::nd_range<3>(gws, lws), gemm1w64Fn);
        });
    }
  }

  if (verify) {
    std::cout << "Validating result with reference..." << std::endl;

    q.memcpy(matrixD.data(), d_d, bytesD).wait();

    // Setup and run reference computation
    std::vector<fp32> matrixD_ref(m * n, std::numeric_limits<fp32>::signaling_NaN());
    gemm_cpu_h(m,
               n,
               k,
               matrixA.data(),
               matrixB.data(),
               matrixC.data(),
               matrixD_ref.data(),
               lda,
               ldb,
               ldc,
               ldd,
               alpha,
               beta);

    compareEqual<fp32>(matrixD.data(), matrixD_ref.data(), m * n);
  }

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (uint32_t w = 0; w < repeat; w++) {
    if (impl == 0)
      q.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<3>(gws, lws), gemm0Fn);
      });
    else if (impl == 1) {
      if (WAVE_SIZE == 32)
        q.submit([&](sycl::handler &cgh) {
          cgh.parallel_for(sycl::nd_range<3>(gws, lws), gemm1w32Fn);
        });
      else
        q.submit([&](sycl::handler &cgh) {
          cgh.parallel_for(sycl::nd_range<3>(gws, lws), gemm1w64Fn);
        });
    }
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  double elapsedTimeMs = time * 1e-6;

  auto gFlops = static_cast<double>(m) * n * (1.0 + 2.0 * k) * 1.0e-9;
  auto tFlopsPerSec = gFlops * repeat / elapsedTimeMs;

  // Echo performance
  std::cout << "BlkM, BlkN, BlkK, "
            << "MatM, MatN, MatK, "
            << "alpha, lda, ldb, "
            << "beta, ldc, ldd, "
            << "elapsedMs, Problem Size(GFlops), TFlops/s" << std::endl;

  std::cout << WMMA_M << ", " << WMMA_N << ", " << WMMA_K << ", " << m << ", " << n
            << ", " << k << ", " << alpha << ", " << lda << ", " << ldb << ", " << beta << ", "
            << ldc << ", " << ldd << ", " << elapsedTimeMs << ", " << gFlops << ", "
            << tFlopsPerSec << std::endl;

  sycl::free(d_a, q);
  sycl::free(d_b, q);
  sycl::free(d_c, q);
  sycl::free(d_d, q);

  std::cout << "Finished!" << std::endl;
}

void Usage(std::string program_name) {
  // Utility function to display argument usage
  std::cout << " Incorrect parameters\n";
  std::cout << " Usage: ";
  std::cout << program_name << "<M> <N> <K> <repeat> <verify>\n\n";
  std::cout
      << "Dense matrix-matrix multiplication: D = alpha * (A * B) + beta * C\n";
  std::cout << "A: M * K, B: K * N, C: M * N, D: M * N\n";
  exit(-1);
}


int main(int argc, char *argv[]) {
  if (argc != 7) {
    Usage(argv[0]);
  }

  const uint32_t impl = atoi(argv[1]);
  const uint32_t m = atoi(argv[2]);
  const uint32_t n = atoi(argv[3]);
  const uint32_t k = atoi(argv[4]);
  const int32_t repeat = atoi(argv[5]);
  const int32_t verify = atoi(argv[6]);
  gemm_wmma(impl, m, n, k, 0.5f, 2.0f, repeat, verify);

  return 0;
}
