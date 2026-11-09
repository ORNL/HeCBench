// Pippenger multi-scalar multiplication on BLS12-381 G1, the same
// operation as sppark's mult_pippenger (Apache-2.0).
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMemcpy hipMemcpy
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemset hipMemset
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuGetLastError hipGetLastError
#define gpuGetErrorString hipGetErrorString
#define gpuSuccess hipSuccess
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#else
#include <cuda_runtime.h>
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemcpy cudaMemcpy
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemset cudaMemset
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuGetLastError cudaGetLastError
#define gpuGetErrorString cudaGetErrorString
#define gpuSuccess cudaSuccess
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#endif

#include "kernels.h"
#include "reference.h"

#define gpuCheck(call)                                                         \
  do {                                                                         \
    auto _e = (call);                                                          \
    if (_e != gpuSuccess) {                                                    \
      printf("[GPU ERROR] %s:%d %s\n", __FILE__, __LINE__,                     \
             gpuGetErrorString(_e));                                           \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

static int div_up(uint32_t n, int block) { return (n + block - 1) / block; }

static void launch(int32_t* digits, const scalar_t* scalars, const point_t* points,
                   uint32_t* counts, uint32_t* offsets, uint32_t* cursor, uint32_t* indices,
                   jac_t* buckets, jac_t* wins, jac_t* out, uint32_t n, sched_t s) {
  const int blocks = div_up(n, kBlock);
  const size_t nslots = s.nwins * s.nbins;
  breakdown<<<blocks, kBlock>>>(digits, scalars, n, s.wbits, s.nwins);
  gpuCheck(gpuMemset(counts, 0, sizeof(uint32_t) * nslots));
  histogram<<<blocks, kBlock>>>(digits, counts, n, s.nwins, s.nbins);
  exclusive_scan<<<1, 1>>>(counts, offsets, nslots);
  gpuCheck(gpuMemcpyAsync(cursor, offsets, sizeof(uint32_t) * nslots, gpuMemcpyDeviceToDevice, 0));
  scatter<<<blocks, kBlock>>>(digits, cursor, indices, n, s.nwins, s.nbins);
  bucket_sum<<<dim3(s.max_digit, s.nwins), kBlock>>>(indices, offsets, counts, points, buckets,
                                                     s.nbins);
  integrate<<<1, s.nwins>>>(buckets, wins, s.nbins, s.max_digit);
  fold<<<1, 1>>>(wins, out, s.nwins, s.wbits);
}

static int run(int lg, int repeat) {
  const uint32_t n = 1u << lg;
  sched_t s = choose_sched(n);
  printf("npoints %u\n", n);
  printf("wbits %d\n", s.wbits);

  point_t* points = (point_t*)malloc(sizeof(point_t) * n);
  scalar_t* scalars = (scalar_t*)malloc(sizeof(scalar_t) * n);
  if (!points || !scalars) {
    printf("host allocation failed\n");
    return 1;
  }
  make_instance(n, 0x1234, points, scalars);
  aff_t ref = reference_msm(points, scalars, n);

  int32_t* d_digits = nullptr;
  scalar_t* d_scalars = nullptr;
  point_t* d_points = nullptr;
  uint32_t* d_counts = nullptr;
  uint32_t* d_offsets = nullptr;
  uint32_t* d_cursor = nullptr;
  uint32_t* d_indices = nullptr;
  jac_t* d_buckets = nullptr;
  jac_t* d_wins = nullptr;
  jac_t* d_out = nullptr;

  const size_t nslots = s.nwins * s.nbins;
  gpuCheck(gpuMalloc(&d_digits, sizeof(int32_t) * s.nwins * (size_t)n));
  gpuCheck(gpuMalloc(&d_scalars, sizeof(scalar_t) * n));
  gpuCheck(gpuMalloc(&d_points, sizeof(point_t) * n));
  gpuCheck(gpuMalloc(&d_counts, sizeof(uint32_t) * nslots));
  gpuCheck(gpuMalloc(&d_offsets, sizeof(uint32_t) * nslots));
  gpuCheck(gpuMalloc(&d_cursor, sizeof(uint32_t) * nslots));
  gpuCheck(gpuMalloc(&d_indices, sizeof(uint32_t) * s.nwins * (size_t)n));
  gpuCheck(gpuMalloc(&d_buckets, sizeof(jac_t) * nslots));
  gpuCheck(gpuMalloc(&d_wins, sizeof(jac_t) * s.nwins));
  gpuCheck(gpuMalloc(&d_out, sizeof(jac_t)));

  gpuCheck(gpuMemcpy(d_scalars, scalars, sizeof(scalar_t) * n, gpuMemcpyHostToDevice));
  gpuCheck(gpuMemcpy(d_points, points, sizeof(point_t) * n, gpuMemcpyHostToDevice));

  launch(d_digits, d_scalars, d_points, d_counts, d_offsets, d_cursor, d_indices, d_buckets, d_wins,
         d_out, n, s);
  gpuCheck(gpuDeviceSynchronize());
  gpuCheck(gpuGetLastError());

  jac_t got;
  gpuCheck(gpuMemcpy(&got, d_out, sizeof(got), gpuMemcpyDeviceToHost));
  int rc = 0;
  if (!aff_equal(ref, got)) {
    aff_t g = jac_to_aff(got);
    printf("FAIL\n");
    print_aff("ref", ref);
    print_aff("got", g);
    rc = 1;
  } else {
    printf("PASS\n");
  }

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    launch(d_digits, d_scalars, d_points, d_counts, d_offsets, d_cursor, d_indices, d_buckets,
           d_wins, d_out, n, s);
  }
  gpuCheck(gpuDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the kernel: %f (us)\n\n", (time * 1e-3) / repeat);

  gpuFree(d_digits);
  gpuFree(d_scalars);
  gpuFree(d_points);
  gpuFree(d_counts);
  gpuFree(d_offsets);
  gpuFree(d_cursor);
  gpuFree(d_indices);
  gpuFree(d_buckets);
  gpuFree(d_wins);
  gpuFree(d_out);
  free(points);
  free(scalars);
  return rc;
}

int main(int argc, char** argv) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
  if (repeat <= 0) {
    printf("repeat must be positive\n");
    return 1;
  }
  if (!msm_self_test()) {
    printf("FAIL\nself-test\n");
    return 1;
  }

  int rc = 0;
  // n <= 192, so sppark's rule selects a 10-bit window. The affine
  // reference inverts once per addition, which limits the checked sizes.
  const int logs[] = {4, 5};
  for (int lg : logs) rc |= run(lg, repeat);
  return rc;
}
