/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2021-2022 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <chrono>
#include <cmath>
#include <iostream>
#include <limits> // std::numeric_limits
#include <string>
#include <vector>
#include <cuda.h>
#include <mma.h>
#ifdef HECBENCH_ENABLE_MPI_REPLICAS
#include <mpi.h>
#endif

typedef half fp16;
typedef float fp32;

#include "reference.h"

using namespace nvcuda;

enum class VerificationStatus { Unsupported = -1, Failed = 0, Passed = 1,
                                Disabled = 2 };

static int report_rank = -1;

static void fatal_exit() {
#ifdef HECBENCH_ENABLE_MPI_REPLICAS
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  exit(EXIT_FAILURE);
}

#ifndef CHECK_CUDA_ERROR
#define CHECK_CUDA_ERROR(status)                                               \
  if (status != cudaSuccess) {                                                 \
    fprintf(stderr, "CUDA error: '%s'(%d) at %s:%d\n",                         \
            cudaGetErrorString(status), status, __FILE__, __LINE__);           \
    fatal_exit();                                                              \
  }
#endif

// Matrix data initialization
template <typename DataT>
__host__ static inline void fill(DataT *mat, uint32_t m, uint32_t n) {
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

// Fragment size
const int WMMA_M = 16;
const int WMMA_N = 16;

// multiples of 16
const int WMMA_K = 16;

// Device warp size
const uint32_t WAVE_SIZE = 32;

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
__global__ void gemm_impl0(const uint32_t m, const uint32_t n, const uint32_t k,
                           fp16 const *__restrict__ a,
                           fp16 const *__restrict__ b,
                           fp32 const *c,
                           fp32 *d, const uint32_t lda, const uint32_t ldb,
                           const uint32_t ldc, const uint32_t ldd,
                           const fp32 alpha, const fp32 beta) {
  // Create frags
  auto fragA = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, fp16,
                              wmma::row_major>();
  auto fragB = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, fp16,
                              wmma::col_major>();
  auto fragC = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, fp32>();
  auto fragAcc = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, fp32>();

  wmma::fill_fragment(fragAcc, 0.0f);

  auto cRow = blockIdx.x * WMMA_M;
  auto cCol = blockIdx.y * WMMA_N;

  // Load the inputs
  for (int n = 0; n < k; n += WMMA_K) {
    // Because the mapping of elements to threads in a warp is opaque,
    // each thread just passes the address of the first element
    wmma::load_matrix_sync(fragA, a + cRow * lda + n, lda);
    wmma::load_matrix_sync(fragB, b + cCol * ldb + n, ldb);

    // Matrix multiply - accumulate using MFMA units
    wmma::mma_sync(fragAcc, fragA, fragB, fragAcc);

    // Fetch C matrix
    wmma::load_matrix_sync(fragC, c + cRow * ldc + cCol, ldc, wmma::mem_row_major);

    // D = alpha * A x B + beta * C
    for (int i = 0; i < fragC.num_elements; ++i) {
      fragC.x[i] = alpha * fragAcc.x[i] + beta * fragC.x[i];
    }
  }

  // Store to D (by a single wave)
  wmma::store_matrix_sync(d + cRow * ldd + cCol, fragC, ldd, wmma::mem_row_major);
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
__global__ void gemm_impl1(const uint32_t m, const uint32_t n, const uint32_t k,
                           fp16 const *__restrict__ a,
                           fp16 const *__restrict__ b,
                           fp32 const *c,
                           fp32 *d, const uint32_t lda, const uint32_t ldb,
                           const uint32_t ldc, const uint32_t ldd,
                           const fp32 alpha, const fp32 beta) {
  // Create frags
  auto fragA = wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, fp16,
                              wmma::row_major>();
  auto fragB = wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, fp16,
                              wmma::col_major>();
  auto fragC = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, fp32>();
  auto fragAcc = wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, fp32>();

  wmma::fill_fragment(fragAcc, 0.0f);

  // Map threadIdx to warpIdx
  auto warpIdx = threadIdx.x / WAVE_SIZE;
  auto warpIdy = threadIdx.y;

  // Target C block
  auto cRow = blockIdx.x * TILE_M + warpIdx * WMMA_M;
  auto cCol = blockIdx.y * TILE_N + warpIdy * WMMA_N;

  // Bounds check
  for (int n = 0; n < k; n += WMMA_K) {
    // Load the inputs
    wmma::load_matrix_sync(fragA, a + (cRow * lda + n), lda);
    wmma::load_matrix_sync(fragB, b + (cCol * ldb + n), ldb);

    // Matrix multiply - accumulate using MFMA units
    wmma::mma_sync(fragAcc, fragA, fragB, fragAcc);
  }

  // Fetch C matrix
  wmma::load_matrix_sync(fragC, c + (cRow * ldc + cCol), ldc,
                         wmma::mem_row_major);

  // D = alpha * A x B + beta * C
  for (int i = 0; i < fragC.num_elements; ++i) {
    fragC.x[i] = alpha * fragAcc.x[i] + beta * fragC.x[i];
  }

  // Store to D
  wmma::store_matrix_sync(d + (cRow * ldd + cCol), fragC, ldd,
                          wmma::mem_row_major);
}

__host__ VerificationStatus gemm_wmma(int impl, uint32_t m, uint32_t n,
                                      uint32_t k, fp32 alpha, fp32 beta,
                                      int32_t repeat, int32_t verify) {
  // Bounds check
  if (impl == 0) {
    if (m < WMMA_M || n < WMMA_N || k < WMMA_K || m % WMMA_M || n % WMMA_N || k % WMMA_K) {
      if (report_rank >= 0) std::cout << "[rank " << report_rank << "] ";
      std::cout << "Unsupported size!\n";
      return VerificationStatus::Unsupported;
    }

  } else {
    if ((m < TILE_M) || n < TILE_N || k < WMMA_K || m % WMMA_M || n % WMMA_N || k % WMMA_K ||
        TILE_M / WMMA_M * 32 * TILE_N / WMMA_N > 1024) {
      if (report_rank >= 0) std::cout << "[rank " << report_rank << "] ";
      std::cout << "Unsupported size!\n";
      return VerificationStatus::Unsupported;
    }
  }

  int lda = k; // row major
  int ldb = k; // col major
  int ldc = n; // row major
  int ldd = ldc;

  std::cout << "Initializing host data..." << std::endl;

  // Initialize input matrices
  std::vector<fp16> matrixA(m * k);
  std::vector<fp16> matrixB(k * n);
  std::vector<fp32> matrixC(m * n);

  // Fill outputs with NaN to catch contamination
  std::vector<fp32> matrixD(
      m * n, std::numeric_limits<fp32>::signaling_NaN());

  fill(matrixA.data(), m, k);
  fill(matrixB.data(), k, n);
  fill(matrixC.data(), m, n);

  std::cout << "Initializing device data..." << std::endl;

  // Allocate and copy device memory
  fp16 *d_a, *d_b;
  fp32 *d_c, *d_d;

  const size_t bytesA = matrixA.size() * sizeof(fp16);
  const size_t bytesB = matrixB.size() * sizeof(fp16);
  const size_t bytesC = matrixC.size() * sizeof(fp32);
  const size_t bytesD = matrixD.size() * sizeof(fp32);

  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_a, bytesA));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_b, bytesB));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_c, bytesC));
  CHECK_CUDA_ERROR(cudaMalloc((void**)&d_d, bytesD));

  CHECK_CUDA_ERROR(
      cudaMemcpy(d_a, matrixA.data(), bytesA, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(
      cudaMemcpy(d_b, matrixB.data(), bytesB, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(
      cudaMemcpy(d_c, matrixC.data(), bytesC, cudaMemcpyHostToDevice));
  CHECK_CUDA_ERROR(
      cudaMemcpy(d_d, matrixD.data(), bytesD, cudaMemcpyHostToDevice));

  std::cout << "Launching GEMM kernel..." << std::endl;
  
  dim3 gridDim(1, 1, 1);
  dim3 blockDim(1, 1, 1);

  if (impl == 0) {
    // e.g. when m = n = 32, the kernel is launched with 4 thread blocks of 32 threads each
    gridDim.x = m / WMMA_M;
    gridDim.y = n / WMMA_N;
    blockDim.x = WAVE_SIZE;
  }
  else {
    // e.g. when m = n = 32, the kernel is launched with 1 thread block of 128 threads
    gridDim.x = m / TILE_M;
    gridDim.y = n / TILE_N;
    blockDim.x = TILE_M / WMMA_M * WAVE_SIZE;
    blockDim.y = TILE_N / WMMA_N;
  }

  // Warmup
  for (int32_t w = 0; w < 30; w++) {
    if (impl == 0)
      gemm_impl0<<<gridDim, blockDim>>>(m, n, k, d_a, d_b, d_c, d_d, lda, ldb, ldc, ldd, alpha, beta);
    else if (impl == 1)
      gemm_impl1<<<gridDim, blockDim>>>(m, n, k, d_a, d_b, d_c, d_d, lda, ldb, ldc, ldd, alpha, beta);
  }

  VerificationStatus verification_status = VerificationStatus::Disabled;
  if (verify) {
    std::cout << "Validating result with reference..." << std::endl;

    // Bring kernel result back to host
    CHECK_CUDA_ERROR(
        cudaMemcpy(matrixD.data(), d_d, bytesD, cudaMemcpyDeviceToHost));

    // Setup and run reference computation
    std::vector<fp32> matrixD_ref(
        m * n, std::numeric_limits<fp32>::signaling_NaN());
    gemm_cpu_h(m, n, k, matrixA.data(), matrixB.data(), matrixC.data(),
               matrixD_ref.data(), lda, ldb, ldc, ldd, alpha, beta);

    bool passed = compareEqual<fp32>(matrixD.data(), matrixD_ref.data(), m * n);
    verification_status = passed ? VerificationStatus::Passed
                                 : VerificationStatus::Failed;
    if (report_rank >= 0) std::cout << "[rank " << report_rank << "] ";
    std::cout << (passed ? "PASSED" : "FAILED") << "\n";
  } else {
    if (report_rank >= 0) std::cout << "[rank " << report_rank << "] ";
    std::cout << "Verification disabled" << std::endl;
  }

  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  auto start = std::chrono::steady_clock::now();

  for (int32_t w = 0; w < repeat; w++) {
    if (impl == 0)
      gemm_impl0<<<gridDim, blockDim>>>(m, n, k, d_a, d_b, d_c, d_d, lda, ldb, ldc, ldd, alpha, beta);
    else if (impl == 1)
      gemm_impl1<<<gridDim, blockDim>>>(m, n, k, d_a, d_b, d_c, d_d, lda, ldb, ldc, ldd, alpha, beta);

  }
  CHECK_CUDA_ERROR(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  double elapsedTimeMs = time * 1e-6;

  auto gFlops = static_cast<double>(m) * n * (1.0 + 2.0 * k) * 1.0e-9;
  auto tFlopsPerSec = gFlops * repeat / static_cast<double>(elapsedTimeMs);

  // Echo performance
  std::cout << "BlkM, BlkN, BlkK, "
            << "MatM, MatN, MatK, "
            << "alpha, lda, ldb, "
            << "beta, ldc, ldd, "
            << "elapsedMs, Problem Size(GFlops), TFlops/s" << std::endl;

  if (report_rank >= 0) std::cout << "[rank " << report_rank << "] ";
  std::cout << WMMA_M << ", " << WMMA_N << ", " << WMMA_K << ", " << m << ", "
            << n << ", " << k << ", " << alpha << ", " << lda << ", " << ldb
            << ", " << beta << ", " << ldc << ", " << ldd << ", "
            << elapsedTimeMs << ", " << gFlops << ", " << tFlopsPerSec
            << std::endl;

  // Release device memory
  CHECK_CUDA_ERROR(cudaFree(d_a));
  CHECK_CUDA_ERROR(cudaFree(d_b));
  CHECK_CUDA_ERROR(cudaFree(d_c));
  CHECK_CUDA_ERROR(cudaFree(d_d));

  std::cout << "Finished!" << std::endl;
  return verification_status;
}

void Usage(std::string program_name) {
  // Utility function to display argument usage
  std::cout << " Incorrect parameters\n";
  std::cout << " Usage: ";
  std::cout << program_name << "<implementation> <M> <N> <K> <repeat> <verify>\n\n";
  std::cout
      << "Dense matrix-matrix multiplication: D = alpha * (A * B) + beta * C\n";
  std::cout << "A: M * K, B: K * N, C: M * N, D: M * N\n";
  fatal_exit();
}

int main(int argc, char *argv[]) {
#ifdef HECBENCH_ENABLE_MPI_REPLICAS
  MPI_Init(&argc, &argv);
  int rank_count, local_rank;
  MPI_Comm local_comm;
  MPI_Comm_rank(MPI_COMM_WORLD, &report_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, report_rank,
                      MPI_INFO_NULL, &local_comm);
  MPI_Comm_rank(local_comm, &local_rank);

  int visible_devices = 0;
  if (cudaGetDeviceCount(&visible_devices) != cudaSuccess || visible_devices < 1) {
    std::cerr << "[rank " << report_rank << "] No visible CUDA device\n";
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  int selected_device = visible_devices == 1 ? 0 : local_rank;
  if (selected_device >= visible_devices) {
    std::cerr << "[rank " << report_rank << "] local rank " << local_rank
              << " cannot be mapped to " << visible_devices
              << " visible CUDA devices\n";
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (cudaSetDevice(selected_device) != cudaSuccess) {
    std::cerr << "[rank " << report_rank << "] Failed to select CUDA device "
              << selected_device << "\n";
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  char processor[MPI_MAX_PROCESSOR_NAME];
  int processor_length;
  MPI_Get_processor_name(processor, &processor_length);
  std::cout << "[rank " << report_rank << "/" << rank_count
            << " local_rank " << local_rank << "] host=" << processor
            << " device=" << selected_device
            << " visible_devices=" << visible_devices << "\n";
  MPI_Comm_free(&local_comm);
#endif

  if (argc != 7) {
    Usage(argv[0]);
  }

  const uint32_t impl = atoi(argv[1]);
  const uint32_t m = atoi(argv[2]);
  const uint32_t n = atoi(argv[3]);
  const uint32_t k = atoi(argv[4]);
  const int32_t repeat = atoi(argv[5]);
  const int32_t verify = atoi(argv[6]);
  [[maybe_unused]] VerificationStatus status =
      gemm_wmma(impl, m, n, k, 0.5f, 2.0f, repeat, verify);

#ifdef HECBENCH_ENABLE_MPI_REPLICAS
  int local_status = static_cast<int>(status);
  int overall_status = 0;
  MPI_Allreduce(&local_status, &overall_status, 1, MPI_INT, MPI_MIN,
                MPI_COMM_WORLD);
  if (report_rank == 0) {
    if (overall_status == static_cast<int>(VerificationStatus::Passed)) {
      std::cout << "OVERALL PASSED (" << rank_count << "/" << rank_count
                << " replica ranks)\n";
    } else if (overall_status ==
               static_cast<int>(VerificationStatus::Disabled)) {
      std::cout << "OVERALL NOT VERIFIED (verification disabled)\n";
    } else {
      std::cout << "OVERALL FAILED\n";
    }
  }
  MPI_Finalize();
  return overall_status > static_cast<int>(VerificationStatus::Failed) ? 0 : 2;
#endif

  return 0;
}
