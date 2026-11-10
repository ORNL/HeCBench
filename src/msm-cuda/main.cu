// Pippenger multi-scalar multiplication on BLS12-381 G1, the same
// operation as sppark's mult_pippenger (Apache-2.0).
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cstdio>
#include <cstdlib>

#if defined(__HIPCC__)
#include <hip/hip_runtime.h>
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMemcpy hipMemcpy
#define gpuMemsetAsync hipMemsetAsync
#define gpuDeviceSynchronize hipDeviceSynchronize
#define gpuGetLastError hipGetLastError
#define gpuGetErrorString hipGetErrorString
#define gpuSuccess hipSuccess
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#else
#include <cuda_runtime.h>
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemcpy cudaMemcpy
#define gpuMemsetAsync cudaMemsetAsync
#define gpuDeviceSynchronize cudaDeviceSynchronize
#define gpuGetLastError cudaGetLastError
#define gpuGetErrorString cudaGetErrorString
#define gpuSuccess cudaSuccess
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
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

static constexpr int kDefaultLog = 20;

static int div_up(uint32_t n, int block) { return (n + block - 1) / block; }

static size_t max_segments(uint32_t n, sched_t s) {
  return (size_t)s.nwins * s.nbins + (size_t)s.nwins * n / kSeg;
}

static jac_t launch(int32_t* digits, const scalar_t* scalars, const point_t* points,
                    uint32_t* counts, uint32_t* offsets, uint32_t* cursor, uint32_t* seg_off,
                    uint32_t* nsegs, uint32_t* indices, jac_t* partials, jac_t* buckets,
                    jac_t* wins, jac_t* h_wins, uint32_t n, sched_t s) {
  const int blocks = div_up(n, kBlock);
  const int nslots = s.nwins * s.nbins;
  breakdown<<<blocks, kBlock>>>(digits, scalars, n, s.wbits, s.nwins);
  gpuCheck(gpuMemsetAsync(counts, 0, sizeof(uint32_t) * nslots, 0));
  histogram<<<blocks, kBlock>>>(digits, counts, n, s.nwins, s.nbins);
  exclusive_scan<<<1, kScan>>>(counts, offsets, cursor, seg_off, nsegs, nslots);
  scatter<<<blocks, kBlock>>>(digits, cursor, indices, n, s.nwins, s.nbins);
  bucket_sum<<<div_up(max_segments(n, s), kBlock), kBlock>>>(indices, offsets, counts, seg_off,
                                                             nsegs, points, partials, nslots);
  bucket_merge<<<div_up(nslots, kBlock), kBlock>>>(partials, counts, seg_off, buckets, nslots);
  bucket_merge_wide<<<nslots, kBlock>>>(partials, counts, seg_off, buckets);
  integrate<<<s.nwins, kBlock>>>(buckets, wins, s.nbins, s.max_digit);
  gpuCheck(gpuMemcpy(h_wins, wins, sizeof(jac_t) * s.nwins, gpuMemcpyDeviceToHost));
  return fold_windows(h_wins, s.nwins, s.wbits);
}

static int run(int lg, int repeat) {
  const uint32_t n = 1u << lg;
  sched_t s = choose_sched(n);
  printf("npoints %u\n", n);
  printf("wbits %d\n", s.wbits);

  point_t* points = (point_t*)malloc(sizeof(point_t) * n);
  scalar_t* scalars = (scalar_t*)malloc(sizeof(scalar_t) * n);
  jac_t* h_wins = (jac_t*)malloc(sizeof(jac_t) * s.nwins);
  if (!points || !scalars || !h_wins) {
    printf("host allocation failed\n");
    free(points);
    free(scalars);
    free(h_wins);
    return 1;
  }
  make_instance(n, 0x1234, points, scalars);
  for (uint32_t i = 0; i < n; i++) {
    if (!on_curve(points[i])) {
      printf("FAIL\npoint %u is not on the curve\n", i);
      free(points);
      free(scalars);
      free(h_wins);
      return 1;
    }
  }
  jac_t ref = pippenger_ref(points, scalars, n);

  int32_t* d_digits = nullptr;
  scalar_t* d_scalars = nullptr;
  point_t* d_points = nullptr;
  uint32_t* d_counts = nullptr;
  uint32_t* d_offsets = nullptr;
  uint32_t* d_cursor = nullptr;
  uint32_t* d_seg_off = nullptr;
  uint32_t* d_nsegs = nullptr;
  uint32_t* d_indices = nullptr;
  jac_t* d_partials = nullptr;
  jac_t* d_buckets = nullptr;
  jac_t* d_wins = nullptr;

  const size_t nslots = s.nwins * s.nbins;
  gpuCheck(gpuMalloc(&d_digits, sizeof(int32_t) * s.nwins * (size_t)n));
  gpuCheck(gpuMalloc(&d_scalars, sizeof(scalar_t) * n));
  gpuCheck(gpuMalloc(&d_points, sizeof(point_t) * n));
  gpuCheck(gpuMalloc(&d_counts, sizeof(uint32_t) * nslots));
  gpuCheck(gpuMalloc(&d_offsets, sizeof(uint32_t) * nslots));
  gpuCheck(gpuMalloc(&d_cursor, sizeof(uint32_t) * nslots));
  gpuCheck(gpuMalloc(&d_seg_off, sizeof(uint32_t) * nslots));
  gpuCheck(gpuMalloc(&d_nsegs, sizeof(uint32_t)));
  gpuCheck(gpuMalloc(&d_indices, sizeof(uint32_t) * s.nwins * (size_t)n));
  gpuCheck(gpuMalloc(&d_partials, sizeof(jac_t) * max_segments(n, s)));
  gpuCheck(gpuMalloc(&d_buckets, sizeof(jac_t) * nslots));
  gpuCheck(gpuMalloc(&d_wins, sizeof(jac_t) * s.nwins));

  gpuCheck(gpuMemcpy(d_scalars, scalars, sizeof(scalar_t) * n, gpuMemcpyHostToDevice));
  gpuCheck(gpuMemcpy(d_points, points, sizeof(point_t) * n, gpuMemcpyHostToDevice));

  jac_t got = launch(d_digits, d_scalars, d_points, d_counts, d_offsets, d_cursor, d_seg_off,
                     d_nsegs, d_indices, d_partials, d_buckets, d_wins, h_wins, n, s);
  gpuCheck(gpuGetLastError());

  int rc = 0;
  if (!jac_equal(ref, got)) {
    printf("FAIL\n");
    print_aff("ref", jac_to_aff(ref));
    print_aff("got", jac_to_aff(got));
    rc = 1;
  } else {
    printf("PASS\n");
  }

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    launch(d_digits, d_scalars, d_points, d_counts, d_offsets, d_cursor, d_seg_off, d_nsegs,
           d_indices, d_partials, d_buckets, d_wins, h_wins, n, s);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the kernel: %f (us)\n\n", (time * 1e-3) / repeat);

  gpuCheck(gpuFree(d_digits));
  gpuCheck(gpuFree(d_scalars));
  gpuCheck(gpuFree(d_points));
  gpuCheck(gpuFree(d_counts));
  gpuCheck(gpuFree(d_offsets));
  gpuCheck(gpuFree(d_cursor));
  gpuCheck(gpuFree(d_seg_off));
  gpuCheck(gpuFree(d_nsegs));
  gpuCheck(gpuFree(d_indices));
  gpuCheck(gpuFree(d_partials));
  gpuCheck(gpuFree(d_buckets));
  gpuCheck(gpuFree(d_wins));
  free(points);
  free(scalars);
  free(h_wins);
  return rc;
}

int main(int argc, char** argv) {
  if (argc != 2 && argc != 3) {
    printf("Usage: %s <repeat> [log2 of the number of points, default %d]\n", argv[0], kDefaultLog);
    return 1;
  }
  const int repeat = atoi(argv[1]);
  const int lg = argc == 3 ? atoi(argv[2]) : kDefaultLog;
  if (repeat <= 0) {
    printf("repeat must be positive\n");
    return 1;
  }
  if (lg < 1 || lg > 24) {
    printf("log2 of the number of points must be in [1, 24]\n");
    return 1;
  }
  if (!msm_self_test()) {
    printf("FAIL\nself-test\n");
    return 1;
  }
  return run(lg, repeat);
}
