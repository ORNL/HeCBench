// Pippenger multi-scalar multiplication on BLS12-381 G1, the same
// operation as sppark's mult_pippenger (Apache-2.0).
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <cstdio>
#include <cstdlib>

#include <sycl/sycl.hpp>

#include "kernels.h"
#include "reference.h"

static int div_up(uint32_t n, int block) { return (n + block - 1) / block; }

static void launch(sycl::queue& q, int32_t* digits, const scalar_t* scalars, const point_t* points,
                   uint32_t* counts, uint32_t* offsets, uint32_t* cursor, uint32_t* indices,
                   jac_t* buckets, jac_t* wins, jac_t* out, uint32_t n, sched_t s) {
  const int blocks = div_up(n, kBlock);
  const size_t nslots = s.nwins * s.nbins;
  q.parallel_for(sycl::nd_range<1>(blocks * kBlock, kBlock), [=](sycl::nd_item<1> it) {
    breakdown(digits, scalars, n, s.wbits, s.nwins, it);
  });
  q.memset(counts, 0, sizeof(uint32_t) * nslots);
  q.parallel_for(sycl::nd_range<1>(blocks * kBlock, kBlock), [=](sycl::nd_item<1> it) {
    histogram(digits, counts, n, s.nwins, s.nbins, it);
  });
  q.single_task([=]() { exclusive_scan(counts, offsets, nslots); });
  q.memcpy(cursor, offsets, sizeof(uint32_t) * nslots);

  q.parallel_for(sycl::nd_range<1>(blocks * kBlock, kBlock), [=](sycl::nd_item<1> it) {
    scatter(digits, cursor, indices, n, s.nwins, s.nbins, it);
  });

  q.submit([&](sycl::handler& h) {
    sycl::local_accessor<jac_t, 1> sm(sycl::range<1>(kBlock), h);
    h.parallel_for(sycl::nd_range<2>(sycl::range<2>(s.nwins, s.max_digit * kBlock),
                                     sycl::range<2>(1, kBlock)),
                   [=](sycl::nd_item<2> it) {
                     bucket_sum(indices, offsets, counts, points, buckets,
                                sm.get_multi_ptr<sycl::access::decorated::no>().get(), s.nbins, it);
                   });
  });

  q.parallel_for(sycl::nd_range<1>(s.nwins, s.nwins), [=](sycl::nd_item<1> it) {
    integrate(buckets, wins, s.nbins, s.max_digit, it);
  });
  q.single_task([=]() { fold(wins, out, s.nwins, s.wbits); });
}

static int run(sycl::queue& q, int lg, int repeat) {
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

  const size_t nslots = s.nwins * s.nbins;
  int32_t* d_digits = sycl::malloc_device<int32_t>(s.nwins * (size_t)n, q);
  scalar_t* d_scalars = sycl::malloc_device<scalar_t>(n, q);
  point_t* d_points = sycl::malloc_device<point_t>(n, q);
  uint32_t* d_counts = sycl::malloc_device<uint32_t>(nslots, q);
  uint32_t* d_offsets = sycl::malloc_device<uint32_t>(nslots, q);
  uint32_t* d_cursor = sycl::malloc_device<uint32_t>(nslots, q);
  uint32_t* d_indices = sycl::malloc_device<uint32_t>(s.nwins * (size_t)n, q);
  jac_t* d_buckets = sycl::malloc_device<jac_t>(nslots, q);
  jac_t* d_wins = sycl::malloc_device<jac_t>(s.nwins, q);
  jac_t* d_out = sycl::malloc_device<jac_t>(1, q);
  if (!d_digits || !d_scalars || !d_points || !d_counts || !d_offsets || !d_cursor || !d_indices ||
      !d_buckets || !d_wins || !d_out) {
    printf("device allocation failed\n");
    return 1;
  }

  q.memcpy(d_scalars, scalars, sizeof(scalar_t) * n);
  q.memcpy(d_points, points, sizeof(point_t) * n);
  q.wait();

  launch(q, d_digits, d_scalars, d_points, d_counts, d_offsets, d_cursor, d_indices, d_buckets,
         d_wins, d_out, n, s);
  jac_t got;
  q.memcpy(&got, d_out, sizeof(got)).wait();

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
    launch(q, d_digits, d_scalars, d_points, d_counts, d_offsets, d_cursor, d_indices, d_buckets,
           d_wins, d_out, n, s);
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the kernel: %f (us)\n\n", (time * 1e-3) / repeat);

  sycl::free(d_digits, q);
  sycl::free(d_scalars, q);
  sycl::free(d_points, q);
  sycl::free(d_counts, q);
  sycl::free(d_offsets, q);
  sycl::free(d_cursor, q);
  sycl::free(d_indices, q);
  sycl::free(d_buckets, q);
  sycl::free(d_wins, q);
  sycl::free(d_out, q);
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

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  int rc = 0;
  // n <= 192, so sppark's rule selects a 10-bit window. The affine
  // reference inverts once per addition, which limits the checked sizes.
  const int logs[] = {4, 5};
  for (int lg : logs) rc |= run(q, lg, repeat);
  return rc;
}
