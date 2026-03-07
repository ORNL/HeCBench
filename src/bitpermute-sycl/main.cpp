// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <sycl/sycl.hpp>
#include "kernels.h"
#include "reference.h"
#include <cstdlib>

static void bit_rev(sycl::queue &q, fr_t* d_out, const fr_t* d_inp, uint32_t lg_domain_size)
{
  size_t domain_size = (size_t)1 << lg_domain_size;
  const uint32_t Z_COUNT = 256 / sizeof(fr_t); // 4: 64, 8: 32
  const uint32_t bsize = Z_COUNT>WARP_SZ ? Z_COUNT : WARP_SZ;

  if (domain_size <= 1024)
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(domain_size),
                                     sycl::range<1>(domain_size)),
      [=](sycl::nd_item<1> item) {
      bit_rev_permutation(d_out, d_inp, lg_domain_size, item);
  });

  else if (domain_size < bsize * Z_COUNT)
    q.parallel_for(
      sycl::nd_range<1>(sycl::range<1>(domain_size),
                        sycl::range<1>(WARP_SZ)),
      [=](sycl::nd_item<1> item) {
       bit_rev_permutation(d_out, d_inp, lg_domain_size, item);
    });

  else if (Z_COUNT > WARP_SZ || lg_domain_size <= 32)
    q.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<uint8_t, 1> sm (
       sycl::range<1>(bsize * Z_COUNT * sizeof(fr_t)), cgh);

   cgh.parallel_for(
       sycl::nd_range<1>(sycl::range<1>(domain_size / Z_COUNT),
                         sycl::range<1>(bsize)),
       [=](sycl::nd_item<1> item) {
        bit_rev_permutation_z(
            d_out, d_inp, lg_domain_size, item,
            sm.get_multi_ptr<sycl::access::decorated::no>().get());
       });
  });
  else {
    int numProcs = q.get_device().get_info<sycl::info::device::max_compute_units>();
    q.submit([&](sycl::handler &cgh) {
     sycl::local_accessor<uint8_t, 1> sm(
         sycl::range<1>(BLOCK_SIZE * Z_COUNT * sizeof(fr_t)), cgh);

     cgh.parallel_for(
         sycl::nd_range<1>(sycl::range<1>(numProcs * 2 * BLOCK_SIZE),
                           sycl::range<1>(BLOCK_SIZE)),
         [=](sycl::nd_item<1> item) {
          bit_rev_permutation_z(
              d_out, d_inp, lg_domain_size, item,
              sm.get_multi_ptr<sycl::access::decorated::no>().get());
         });
     });
  }
}

void bit_permute(sycl::queue &q, const int lg_domain_size, const int repeat)
{
  assert(lg_domain_size <= MAX_LG_DOMAIN_SIZE);
  size_t domain_size = (size_t)1 << lg_domain_size;
  size_t domain_size_bytes = sizeof(fr_t) *  domain_size;

  printf("Domain size is %zu\n", domain_size);

  fr_t *d_inout = sycl::malloc_device<fr_t>(domain_size, q);
  fr_t *inout = (fr_t*) malloc (domain_size_bytes);
  fr_t *out = (fr_t*) malloc (domain_size_bytes);
  #pragma omp parallel for
  for (size_t i = 0; i < domain_size; i++) {
    out[i] = inout[i] = i;
  }

  // warmup and verify
  q.memcpy(d_inout, inout, domain_size_bytes);
  bit_rev(q, d_inout, d_inout, lg_domain_size);
  bit_rev_cpu(out, inout, lg_domain_size);
  q.memcpy(inout, d_inout, domain_size_bytes).wait();
  int error = memcmp(out, inout, domain_size_bytes);
  printf("%s\n", error ? "FAIL" : "PASS");
  if (error) exit(1);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    bit_rev(q, d_inout, d_inout, lg_domain_size);

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the kernel: %f (us)\n\n", (time * 1e-3f) / repeat);

  sycl::free(d_inout, q);
  free(inout);
  free(out);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  bit_permute(q, 10, repeat);
  bit_permute(q, 11, repeat);
  bit_permute(q, 15, repeat);
  bit_permute(q, 27, repeat);
  bit_permute(q, 28, repeat);
}
