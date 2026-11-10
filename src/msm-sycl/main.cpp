// Pippenger multi-scalar multiplication on BLS12-381 G1, the same
// operation as sppark's mult_pippenger (Apache-2.0).
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cstdio>
#include <cstdlib>

#include <sycl/sycl.hpp>

#include "kernels.h"
#include "reference.h"

static constexpr int kDefaultLog = 20;

static int div_up(uint32_t n, int block) { return (n + block - 1) / block; }

static size_t max_segments(uint32_t n, sched_t s) {
  return (size_t)s.nwins * s.nbins + (size_t)s.nwins * n / kSeg;
}

static jac_t launch(sycl::queue& q, int32_t* digits, const scalar_t* scalars, const point_t* points,
                    uint32_t* counts, uint32_t* offsets, uint32_t* cursor, uint32_t* seg_off,
                    uint32_t* nsegs, uint32_t* indices, jac_t* partials, jac_t* buckets,
                    jac_t* wins, jac_t* h_wins, uint32_t n, sched_t s) {
  const int blocks = div_up(n, kBlock);
  const int nslots = s.nwins * s.nbins;
  q.parallel_for(sycl::nd_range<1>(blocks * kBlock, kBlock), [=](sycl::nd_item<1> it) {
    breakdown(digits, scalars, n, s.wbits, s.nwins, it);
  });
  q.memset(counts, 0, sizeof(uint32_t) * nslots);
  q.parallel_for(sycl::nd_range<1>(blocks * kBlock, kBlock), [=](sycl::nd_item<1> it) {
    histogram(digits, counts, n, s.nwins, s.nbins, it);
  });
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<uint32_t, 1> sm(sycl::range<1>(kScan), h);
    h.parallel_for(sycl::nd_range<1>(kScan, kScan), [=](sycl::nd_item<1> it) {
      exclusive_scan(counts, offsets, cursor, seg_off, nsegs, nslots,
                     sm.get_multi_ptr<sycl::access::decorated::no>().get(), it);
    });
  });
  q.parallel_for(sycl::nd_range<1>(blocks * kBlock, kBlock), [=](sycl::nd_item<1> it) {
    scatter(digits, cursor, indices, n, s.nwins, s.nbins, it);
  });
  const size_t seg_groups = div_up(max_segments(n, s), kBlock);
  q.parallel_for(sycl::nd_range<1>(seg_groups * kBlock, kBlock), [=](sycl::nd_item<1> it) {
    bucket_sum(indices, offsets, counts, seg_off, nsegs, points, partials, nslots, it);
  });
  q.parallel_for(sycl::nd_range<1>(div_up(nslots, kBlock) * kBlock, kBlock),
                 [=](sycl::nd_item<1> it) {
                   bucket_merge(partials, counts, seg_off, buckets, nslots, it);
                 });
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<jac_t, 1> sm(sycl::range<1>(kBlock), h);
    h.parallel_for(sycl::nd_range<1>((size_t)nslots * kBlock, kBlock), [=](sycl::nd_item<1> it) {
      bucket_merge_wide(partials, counts, seg_off, buckets,
                        sm.get_multi_ptr<sycl::access::decorated::no>().get(), it);
    });
  });
  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<jac_t, 1> sm(sycl::range<1>(kBlock), h);
    h.parallel_for(sycl::nd_range<1>(s.nwins * kBlock, kBlock), [=](sycl::nd_item<1> it) {
      integrate(buckets, wins, s.nbins, s.max_digit,
                sm.get_multi_ptr<sycl::access::decorated::no>().get(), it);
    });
  });
  q.memcpy(h_wins, wins, sizeof(jac_t) * s.nwins).wait();
  return fold_windows(h_wins, s.nwins, s.wbits);
}

static int run(sycl::queue& q, int lg, int repeat) {
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

  const size_t nslots = s.nwins * s.nbins;
  int32_t* d_digits = sycl::malloc_device<int32_t>(s.nwins * (size_t)n, q);
  scalar_t* d_scalars = sycl::malloc_device<scalar_t>(n, q);
  point_t* d_points = sycl::malloc_device<point_t>(n, q);
  uint32_t* d_counts = sycl::malloc_device<uint32_t>(nslots, q);
  uint32_t* d_offsets = sycl::malloc_device<uint32_t>(nslots, q);
  uint32_t* d_cursor = sycl::malloc_device<uint32_t>(nslots, q);
  uint32_t* d_seg_off = sycl::malloc_device<uint32_t>(nslots, q);
  uint32_t* d_nsegs = sycl::malloc_device<uint32_t>(1, q);
  uint32_t* d_indices = sycl::malloc_device<uint32_t>(s.nwins * (size_t)n, q);
  jac_t* d_partials = sycl::malloc_device<jac_t>(max_segments(n, s), q);
  jac_t* d_buckets = sycl::malloc_device<jac_t>(nslots, q);
  jac_t* d_wins = sycl::malloc_device<jac_t>(s.nwins, q);

  auto release = [&]() {
    sycl::free(d_digits, q);
    sycl::free(d_scalars, q);
    sycl::free(d_points, q);
    sycl::free(d_counts, q);
    sycl::free(d_offsets, q);
    sycl::free(d_cursor, q);
    sycl::free(d_seg_off, q);
    sycl::free(d_nsegs, q);
    sycl::free(d_indices, q);
    sycl::free(d_partials, q);
    sycl::free(d_buckets, q);
    sycl::free(d_wins, q);
    free(points);
    free(scalars);
    free(h_wins);
  };

  if (!d_digits || !d_scalars || !d_points || !d_counts || !d_offsets || !d_cursor || !d_seg_off ||
      !d_nsegs || !d_indices || !d_partials || !d_buckets || !d_wins) {
    printf("device allocation failed\n");
    release();
    return 1;
  }

  q.memcpy(d_scalars, scalars, sizeof(scalar_t) * n);
  q.memcpy(d_points, points, sizeof(point_t) * n);
  q.wait();

  jac_t got = launch(q, d_digits, d_scalars, d_points, d_counts, d_offsets, d_cursor, d_seg_off,
                     d_nsegs, d_indices, d_partials, d_buckets, d_wins, h_wins, n, s);

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
    launch(q, d_digits, d_scalars, d_points, d_counts, d_offsets, d_cursor, d_seg_off, d_nsegs,
           d_indices, d_partials, d_buckets, d_wins, h_wins, n, s);
  }
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the kernel: %f (us)\n\n", (time * 1e-3) / repeat);

  release();
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

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  return run(q, lg, repeat);
}
