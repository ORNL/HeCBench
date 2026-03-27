#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <sycl/sycl.hpp>
#include "tensorAccessor.h"

// A demo of packed tensor accessors in Pytorch
void tensor_packed_accessor_kernel (
    sycl::nd_item<1> &item,
    PackedTensorAccessor64<float, 1, RestrictPtrTraits> r,
    PackedTensorAccessor64<float, 2, RestrictPtrTraits> m,
    PackedTensorAccessor64<float, 1, RestrictPtrTraits> v)
{
  int64_t i = item.get_global_id(0);
  if (i < r.size(0)) {
    float val = 0.0f;
    for (int64_t j = 0; j < m.size(1); j++) {
      val += m[i][j] * v[j];
    }
    r[i] = val;
  }
}

void raw_accessor_kernel (
    sycl::nd_item<1> &item,
    const int64_t nrow,
    const int64_t ncol,
          float *__restrict__ r,
    const float *__restrict__ m,
    const float *__restrict__ v)
{
  int64_t i = item.get_global_id(0);
  if (i < nrow) {
    float val = 0.0f;
    for (int64_t j = 0; j < ncol; j++) {
      val += m[i * ncol + j] * v[j];
    }
    r[i] = val;
  }
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <number of rows> <number of columns> <repeat>\n", argv[0]);
    return 1;
  }

  const int64_t nrow = atol(argv[1]);
  const int64_t ncol = atol(argv[2]);
  const int repeat = atoi(argv[3]);

  // tensor sizes and strides
  const int64_t sizes[2] = {nrow, ncol};
  const int64_t strides[2] = {ncol, 1};

  int64_t numel = 1;
  for (int i = 0; i < 2; i++) numel *= sizes[i];

  // matrix vector multiply
  int64_t m_bytes = numel * sizeof(float);
  int64_t v_bytes = ncol * sizeof(float);
  int64_t r_bytes = nrow * sizeof(float);

  float *m, *v, *r, *r_ref;
  m = (float*) malloc (m_bytes);
  v = (float*) malloc (v_bytes);
  r = (float*) malloc (r_bytes);
  r_ref = (float*) malloc (r_bytes);

  srand(123);
  for (int64_t i = 0; i < numel; i++) {
    m[i] = rand() / (float)RAND_MAX;
  }

  for (int64_t i = 0; i < ncol; i++) {
    v[i] = rand() / (float)RAND_MAX;
  }

  for (int64_t i = 0; i < nrow; i++) {
    float val = 0.f;
    for (int64_t j = 0; j < ncol; j++) {
      val += m[i * ncol + j] * v[j];
    }
    r_ref[i] = val;
  }

  // Note: signficant performance drop using an out-of-order queue
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_m, *d_v, *d_r;
  d_m = sycl::malloc_device<float>(numel, q);
  q.memcpy(d_m, m, m_bytes);

  d_v = sycl::malloc_device<float>(ncol, q);
  q.memcpy(d_v, v, v_bytes);

  d_r = sycl::malloc_device<float>(nrow, q);

  PackedTensorAccessor64<float, 2, RestrictPtrTraits> m_acc (d_m, sizes, strides);
  PackedTensorAccessor64<float, 1, RestrictPtrTraits> v_acc (d_v, &ncol, strides+1);
  PackedTensorAccessor64<float, 1, RestrictPtrTraits> r_acc (d_r, &nrow, strides+1);

  sycl::range<1> gws ((nrow + 255) / 256 * 256);
  sycl::range<1> lws (256);

  auto taFn = [=] (sycl::nd_item<1> item) {
    tensor_packed_accessor_kernel(item, r_acc, m_acc, v_acc);
  };
  auto raFn = [=] (sycl::nd_item<1> item) {
    raw_accessor_kernel(item, nrow, ncol, d_r, d_m, d_v);
  };

  printf("Warmup..\n");
  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), taFn);
    });
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), raFn);
    });
  }

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), raFn);
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of raw_accessor_kernel: %f (us)\n",
          time * 1e-3f / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for(sycl::nd_range<1>(gws, lws), taFn);
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of tensor_packed_accessor_kernel: %f (us)\n",
          time * 1e-3f / repeat);

  q.memcpy(r, d_r, r_bytes).wait();

  sycl::free(d_m, q);
  sycl::free(d_v, q);
  sycl::free(d_r, q);

  // verify (may fail due to floating-point rounding)
  bool ok = true;
  for (int64_t i = 0; i < nrow; i++) {
    if (fabsf(r[i] - r_ref[i]) > 1e-3f) {
      printf("%f %f\n", r[i], r_ref[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(m);
  free(v);
  free(r);
  free(r_ref);

  return 0;
}
