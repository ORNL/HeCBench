#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <vector>
#include <hip/hip_runtime.h>
#include "reference.h"

#define CHECK(call)                                                          \
  do {                                                                       \
    const hipError_t err = (call);                                           \
    if (err != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error %s:%d '%s': %s\n", __FILE__, __LINE__,      \
              #call, hipGetErrorString(err));                               \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

// Original HiOp mapping is one thread per row i, looping over all partner rows
// j -- only O(m) work items, which leaves the GPU underused. Here each thread
// instead owns a single (i,j) row pair and performs one merge, raising the
// parallelism to O(m^2). Threads are laid out 2D with the fast (x) dimension on
// j so the writes to W[i][j] are coalesced; BLOCK_X is warp/wavefront-aligned.
#define BLOCK_X 32
#define BLOCK_Y 8

// W += alpha * J * D^{-1} * J^T : thread (i,j) computes one upper-triangle entry
// (j >= i) of the symmetric diagonal block.
__global__ void mdinvmtrans_diag(int nrows,
                                 const int* __restrict__ row_start,
                                 const int* __restrict__ jcol,
                                 const double* __restrict__ values,
                                 const double* __restrict__ D,
                                 int row_dest_start, int col_dest_start,
                                 double alpha, double* __restrict__ W, int m_W)
{
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= nrows || j >= nrows || j < i) return;

  double acc = 0.0;
  if (i == j) {
    for (int k = row_start[i]; k < row_start[i + 1]; k++)
      acc += values[k] / D[jcol[k]] * values[k];
  } else {
    int ki = row_start[i], kj = row_start[j];
    const int kie = row_start[i + 1], kje = row_start[j + 1];
    while (ki < kie && kj < kje) {
      const int ci = jcol[ki], cj = jcol[kj];
      if (ci == cj) { acc += values[ki] / D[ci] * values[kj]; ki++; kj++; }
      else if (ci < cj) ki++;
      else kj++;
    }
  }
  W[(size_t)(i + row_dest_start) * m_W + j + col_dest_start] += alpha * acc;
}

// W += alpha * J1 * D^{-1} * J2^T : thread (i,j) computes one entry of the full
// off-diagonal block by merging row i of J1 against row j of J2.
__global__ void mdinvntrans(int m1, int m2,
                            const int* __restrict__ rs1, const int* __restrict__ jc1,
                            const double* __restrict__ v1,
                            const int* __restrict__ rs2, const int* __restrict__ jc2,
                            const double* __restrict__ v2,
                            const double* __restrict__ D,
                            int row_dest_start, int col_dest_start,
                            double alpha, double* __restrict__ W, int m_W)
{
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int i = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= m1 || j >= m2) return;

  double acc = 0.0;
  int ki = rs1[i], kj = rs2[j];
  const int kie = rs1[i + 1], kje = rs2[j + 1];
  while (ki < kie && kj < kje) {
    const int ci = jc1[ki], cj = jc2[kj];
    if (ci == cj) { acc += v1[ki] / D[ci] * v2[kj]; ki++; kj++; }
    else if (ci < cj) ki++;
    else kj++;
  }
  W[(size_t)(i + row_dest_start) * m_W + j + col_dest_start] += alpha * acc;
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <rows> <nnz per row> <repeat>\n", argv[0]);
    return 1;
  }

  const int m = atoi(argv[1]);
  const int nnz_row = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const int nx = 8 * nnz_row + 1024;
  const double alpha = -1.0;

  std::vector<int> h_rs1, h_jc1, h_rs2, h_jc2;
  std::vector<double> h_v1, h_v2, h_D;
  gen_csr(m, nx, nnz_row, 123, h_rs1, h_jc1, h_v1);
  gen_csr(m, nx, nnz_row, 456, h_rs2, h_jc2, h_v2);
  gen_diag(nx, 789, h_D);

  const int nnz = (int)h_v1.size();
  const size_t w_elems = (size_t)m * m;
  const size_t w_bytes = w_elems * sizeof(double);

  int *d_rs1, *d_jc1, *d_rs2, *d_jc2;
  double *d_v1, *d_v2, *d_D, *d_W;
  CHECK(hipMalloc((void**)&d_rs1, (m + 1) * sizeof(int)));
  CHECK(hipMalloc((void**)&d_jc1, nnz * sizeof(int)));
  CHECK(hipMalloc((void**)&d_v1, nnz * sizeof(double)));
  CHECK(hipMalloc((void**)&d_rs2, (m + 1) * sizeof(int)));
  CHECK(hipMalloc((void**)&d_jc2, nnz * sizeof(int)));
  CHECK(hipMalloc((void**)&d_v2, nnz * sizeof(double)));
  CHECK(hipMalloc((void**)&d_D, nx * sizeof(double)));
  CHECK(hipMalloc((void**)&d_W, w_bytes));

  CHECK(hipMemcpy(d_rs1, h_rs1.data(), (m + 1) * sizeof(int), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_jc1, h_jc1.data(), nnz * sizeof(int), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_v1, h_v1.data(), nnz * sizeof(double), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_rs2, h_rs2.data(), (m + 1) * sizeof(int), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_jc2, h_jc2.data(), nnz * sizeof(int), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_v2, h_v2.data(), nnz * sizeof(double), hipMemcpyHostToDevice));
  CHECK(hipMemcpy(d_D, h_D.data(), nx * sizeof(double), hipMemcpyHostToDevice));

  const dim3 block(BLOCK_X, BLOCK_Y);
  const dim3 grid((m + BLOCK_X - 1) / BLOCK_X, (m + BLOCK_Y - 1) / BLOCK_Y);

  std::vector<double> h_W(w_elems);
  std::vector<double> h_ref(w_elems);
  int errors = 0;

  // --- diagonal block: W += alpha * J D^{-1} J^T ---------------------------

  // host/device correctness check (run once, verify against reference) before timing
  CHECK(hipMemset(d_W, 0, w_bytes));
  mdinvmtrans_diag<<<grid, block>>>(m, d_rs1, d_jc1, d_v1, d_D, 0, 0, alpha, d_W, m);
  CHECK(hipMemcpy(h_W.data(), d_W, w_bytes, hipMemcpyDeviceToHost));

  std::fill(h_ref.begin(), h_ref.end(), 0.0);
  reference_mdinvmtrans_diag(m, m, h_rs1.data(), h_jc1.data(), h_v1.data(), h_D.data(),
                             0, 0, alpha, h_ref.data());
  for (int i = 0; i < m; i++)
    for (int j = i; j < m; j++)
      if (!close_enough(h_W[(size_t)i * m + j], h_ref[(size_t)i * m + j], 1e-10)) {
        errors++; i = m; break;
      }

  // benchmark
  CHECK(hipMemset(d_W, 0, w_bytes));
  CHECK(hipDeviceSynchronize());
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    mdinvmtrans_diag<<<grid, block>>>(
        m, d_rs1, d_jc1, d_v1, d_D, 0, 0, alpha, d_W, m);

  CHECK(hipDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of addMDinvMtrans (diag block): %f (us)\n",
         time * 1e-3 / repeat);

  // --- off-diagonal block: W += alpha * J1 D^{-1} J2^T ---------------------

  // host/device correctness check before timing
  CHECK(hipMemset(d_W, 0, w_bytes));
  mdinvntrans<<<grid, block>>>(m, m, d_rs1, d_jc1, d_v1, d_rs2, d_jc2, d_v2, d_D, 0, 0, alpha, d_W, m);
  CHECK(hipMemcpy(h_W.data(), d_W, w_bytes, hipMemcpyDeviceToHost));

  std::fill(h_ref.begin(), h_ref.end(), 0.0);
  reference_mdinvntrans(m, m, m, h_rs1.data(), h_jc1.data(), h_v1.data(),
                        h_rs2.data(), h_jc2.data(), h_v2.data(), h_D.data(),
                        0, 0, alpha, h_ref.data());
  for (size_t k = 0; k < w_elems; k++)
    if (!close_enough(h_W[k], h_ref[k], 1e-10)) { errors++; break; }

  // benchmark
  CHECK(hipMemset(d_W, 0, w_bytes));
  CHECK(hipDeviceSynchronize());
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    mdinvntrans<<<grid, block>>>(
        m, m, d_rs1, d_jc1, d_v1, d_rs2, d_jc2, d_v2, d_D, 0, 0, alpha, d_W, m);

  CHECK(hipDeviceSynchronize());
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of addMDinvNtrans (off-diag):   %f (us)\n",
         time * 1e-3 / repeat);

  printf("%s\n", errors == 0 ? "PASS" : "FAIL");

  CHECK(hipFree(d_rs1)); CHECK(hipFree(d_jc1)); CHECK(hipFree(d_v1));
  CHECK(hipFree(d_rs2)); CHECK(hipFree(d_jc2)); CHECK(hipFree(d_v2));
  CHECK(hipFree(d_D)); CHECK(hipFree(d_W));

  return 0;
}
