#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <new>
#include <vector>
#include <sycl/sycl.hpp>
#include "reference.h"

// Original HiOp mapping is one thread per row i, looping over all partner rows
// j -- only O(m) work items, which leaves the GPU underused. Here each thread
// instead owns a single (i,j) row pair and performs one merge, raising the
// parallelism to O(m^2). Threads are laid out 2D with the fast (x) dimension on
// j so the writes to W[i][j] are coalesced; BLOCK_X is warp-aligned.
#define BLOCK_X 32
#define BLOCK_Y 8

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <rows> <nnz per row> <repeat>\n", argv[0]);
    return 1;
  }

  const int m = atoi(argv[1]);
  const int nnz_row = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  if (!valid_problem_size(m, nnz_row, repeat)) {
    printf("Invalid arguments: <rows>, <nnz per row> and <repeat> must be "
           "positive, and the number of nonzeros must fit in a 32-bit int\n");
    return 1;
  }

  const int nx = 8 * nnz_row + 1024;
  const double alpha = -1.0;

  const size_t w_elems = (size_t)m * m;
  const size_t w_bytes = w_elems * sizeof(double);

  std::vector<int> h_rs1, h_jc1, h_rs2, h_jc2;
  std::vector<double> h_v1, h_v2, h_D, h_W, h_ref;
  try {
    gen_csr(m, nx, nnz_row, 123, h_rs1, h_jc1, h_v1);
    gen_csr(m, nx, nnz_row, 456, h_rs2, h_jc2, h_v2);
    gen_diag(nx, 789, h_D);
    h_W.resize(w_elems);
    h_ref.resize(w_elems);
  } catch (const std::bad_alloc&) {
    printf("Failed to allocate the host buffers: the dense block alone needs "
           "%zu bytes\n", w_bytes);
    return 1;
  }

  const int nnz = (int)h_v1.size();

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  int *d_rs1 = sycl::malloc_device<int>(m + 1, q);
  int *d_jc1 = sycl::malloc_device<int>(nnz, q);
  double *d_v1 = sycl::malloc_device<double>(nnz, q);
  int *d_rs2 = sycl::malloc_device<int>(m + 1, q);
  int *d_jc2 = sycl::malloc_device<int>(nnz, q);
  double *d_v2 = sycl::malloc_device<double>(nnz, q);
  double *d_D = sycl::malloc_device<double>(nx, q);
  double *d_W = sycl::malloc_device<double>(w_elems, q);

  if (d_rs1 == nullptr || d_jc1 == nullptr || d_v1 == nullptr ||
      d_rs2 == nullptr || d_jc2 == nullptr || d_v2 == nullptr ||
      d_D == nullptr || d_W == nullptr) {
    printf("Failed to allocate the device buffers: the dense block alone needs "
           "%zu bytes\n", w_bytes);
    return 1;
  }

  q.memcpy(d_rs1, h_rs1.data(), (m + 1) * sizeof(int));
  q.memcpy(d_jc1, h_jc1.data(), nnz * sizeof(int));
  q.memcpy(d_v1, h_v1.data(), nnz * sizeof(double));
  q.memcpy(d_rs2, h_rs2.data(), (m + 1) * sizeof(int));
  q.memcpy(d_jc2, h_jc2.data(), nnz * sizeof(int));
  q.memcpy(d_v2, h_v2.data(), nnz * sizeof(double));
  q.memcpy(d_D, h_D.data(), nx * sizeof(double));
  q.wait();

  const int m_W = m;
  const int row_dest_start = 0, col_dest_start = 0;

  const int BX = BLOCK_X, BY = BLOCK_Y;
  const size_t gx = ((size_t)m + BX - 1) / BX * BX;
  const size_t gy = ((size_t)m + BY - 1) / BY * BY;
  const sycl::range<2> local(BY, BX);
  const sycl::range<2> glob(gy, gx);

  int errors = 0;

  // W += alpha * J * D^{-1} * J^T : work-item (i,j) merges row i against row j
  // and writes one entry of the upper triangle of the diagonal block.
  auto run_mdinvmtrans_diag = [&]() {
    q.parallel_for(sycl::nd_range<2>(glob, local), [=](sycl::nd_item<2> item) {
      const int i = item.get_global_id(0);
      const int j = item.get_global_id(1);
      if (i >= m || j >= m || j < i) return;
      double acc = 0.0;
      if (i == j) {
        for (int k = d_rs1[i]; k < d_rs1[i + 1]; k++)
          acc += d_v1[k] / d_D[d_jc1[k]] * d_v1[k];
      } else {
        int ki = d_rs1[i], kj = d_rs1[j];
        const int kie = d_rs1[i + 1], kje = d_rs1[j + 1];
        while (ki < kie && kj < kje) {
          const int ci = d_jc1[ki], cj = d_jc1[kj];
          if (ci == cj) { acc += d_v1[ki] / d_D[ci] * d_v1[kj]; ki++; kj++; }
          else if (ci < cj) ki++;
          else kj++;
        }
      }
      d_W[(size_t)(i + row_dest_start) * m_W + j + col_dest_start] += alpha * acc;
    });
  };

  // W += alpha * J1 * D^{-1} * J2^T : work-item (i,j) merges row i of J1
  // against row j of J2 and writes one entry of the off-diagonal block.
  auto run_mdinvntrans = [&]() {
    q.parallel_for(sycl::nd_range<2>(glob, local), [=](sycl::nd_item<2> item) {
      const int i = item.get_global_id(0);
      const int j = item.get_global_id(1);
      if (i >= m || j >= m) return;
      double acc = 0.0;
      int ki = d_rs1[i], kj = d_rs2[j];
      const int kie = d_rs1[i + 1], kje = d_rs2[j + 1];
      while (ki < kie && kj < kje) {
        const int ci = d_jc1[ki], cj = d_jc2[kj];
        if (ci == cj) { acc += d_v1[ki] / d_D[ci] * d_v2[kj]; ki++; kj++; }
        else if (ci < cj) ki++;
        else kj++;
      }
      d_W[(size_t)(i + row_dest_start) * m_W + j + col_dest_start] += alpha * acc;
    });
  };

  // --- diagonal block: W += alpha * J D^{-1} J^T ---------------------------

  // host/device correctness check (run once, verify against reference) before timing
  q.memset(d_W, 0, w_bytes).wait();
  run_mdinvmtrans_diag();
  q.memcpy(h_W.data(), d_W, w_bytes).wait();

  std::fill(h_ref.begin(), h_ref.end(), 0.0);
  reference_mdinvmtrans_diag(m, m, h_rs1.data(), h_jc1.data(), h_v1.data(), h_D.data(),
                             0, 0, alpha, h_ref.data());
  for (int i = 0; i < m; i++)
    for (int j = i; j < m; j++)
      if (!close_enough(h_W[(size_t)i * m + j], h_ref[(size_t)i * m + j], 1e-10)) {
        errors++; i = m; break;
      }

  // benchmark
  q.memset(d_W, 0, w_bytes).wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) run_mdinvmtrans_diag();
  q.wait();

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of addMDinvMtrans (diag block): %f (us)\n",
         time * 1e-3 / repeat);

  // --- off-diagonal block: W += alpha * J1 D^{-1} J2^T ---------------------

  // host/device correctness check before timing
  q.memset(d_W, 0, w_bytes).wait();
  run_mdinvntrans();
  q.memcpy(h_W.data(), d_W, w_bytes).wait();

  std::fill(h_ref.begin(), h_ref.end(), 0.0);
  reference_mdinvntrans(m, m, m, h_rs1.data(), h_jc1.data(), h_v1.data(),
                        h_rs2.data(), h_jc2.data(), h_v2.data(), h_D.data(),
                        0, 0, alpha, h_ref.data());
  for (size_t k = 0; k < w_elems; k++)
    if (!close_enough(h_W[k], h_ref[k], 1e-10)) { errors++; break; }

  // benchmark
  q.memset(d_W, 0, w_bytes).wait();
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) run_mdinvntrans();
  q.wait();

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of addMDinvNtrans (off-diag):   %f (us)\n",
         time * 1e-3 / repeat);

  printf("%s\n", errors == 0 ? "PASS" : "FAIL");

  sycl::free(d_rs1, q); sycl::free(d_jc1, q); sycl::free(d_v1, q);
  sycl::free(d_rs2, q); sycl::free(d_jc2, q); sycl::free(d_v2, q);
  sycl::free(d_D, q); sycl::free(d_W, q);

  return 0;
}
