#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <chrono>
#include <random>
#include <sycl/sycl.hpp>
#include "reference.h"

void zero_point (
    sycl::nd_item<1> &item,
    const float* x_min,
    const float* x_max,
    int32_t qmin,
    int32_t qmax,
    int size,
    bool preserve_sparsity,
    float* scale,
    int32_t* zero_point)
{
  int i = item.get_global_id(0);
  if (i < size) {
    float min_val = x_min[i];
    float max_val = x_max[i];

    if (min_val < 0 && max_val > 0 && preserve_sparsity) {
      int symmetric_qmin = -((qmax - qmin) / 2 + 1);
      int symmetric_qmax = (qmax - qmin) / 2;
      double max_scale = sycl::fmax(
          sycl::fabs(min_val / static_cast<double>(symmetric_qmin)), sycl::fabs(max_val / static_cast<double>(symmetric_qmin)));
      min_val = max_scale * symmetric_qmin;
      max_val = max_scale * symmetric_qmax;
    }

    // We extend the [min, max] interval to ensure that it contains 0.
    // Otherwise, we would not meet the requirement that 0 be an exactly
    // representable value.
    min_val = sycl::fmin(min_val, 0.f);
    max_val = sycl::fmax(max_val, 0.f);
    scale[i] = (static_cast<double>(max_val) - min_val) / (qmax - qmin);

    // Moving this check outside this function would result in extra Device to
    // Host copy of the min and max val which would result in a perf hit.
    if (scale[i] == 0.0f || sycl::isinf(1.0f / scale[i])) {
      scale[i] = 0.1;
    }

    double zero_point_from_min = qmin - min_val / static_cast<double>(scale[i]);
    double zero_point_from_max = qmax - max_val / static_cast<double>(scale[i]);
    double zero_point_from_min_error = sycl::abs(qmin) + sycl::fabs(min_val / static_cast<double>(scale[i]));
    double zero_point_from_max_error = sycl::abs(qmax) + sycl::fabs(max_val / static_cast<double>(scale[i]));
    double initial_zero_point = zero_point_from_min_error < zero_point_from_max_error
                                ? zero_point_from_min
                                : zero_point_from_max;

    // Note: preserve_sparsity here means symmetric quantization.
    // for symmetric quantization, we force zero_point
    // to be a middle value between qmin and qmax.
    // If either min or max is 0, then we just use 0 as zero_point.
    if (min_val < 0 && max_val > 0 && preserve_sparsity) {
      initial_zero_point = static_cast<double>(qmin + qmax) / 2;
    }
    // Now we need to nudge the zero point to be an integer
    // (our zero points are integer, and this is motivated by the
    // requirement to be able to represent the real value "0" exactly as a
    // quantized value, which is required in multiple places, for example in
    // Im2col with zero padding).
    int32_t nudged_zero_point = 0;
    if (initial_zero_point < qmin) {
      nudged_zero_point = qmin;
    } else if (initial_zero_point > qmax) {
      nudged_zero_point = qmax;
    } else {
      nudged_zero_point = nearbyint(initial_zero_point);
    }
    zero_point[i] = nudged_zero_point;
  }
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of min/max values> <repeat>\n", argv[0]);
    return 1;
  }
  const int size = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  int32_t qmin = -127;
  int32_t qmax =  127;
  bool preserve_sparsity = true;

  size_t size_bytes = sizeof(float) * size;

  float *scale = (float*) malloc (size_bytes);
  float *scale_ref = (float*) malloc (size_bytes);
  int32_t *zp = (int32_t*) malloc (size_bytes);
  int32_t *zp_ref = (int32_t*) malloc (size_bytes);
  float *min = (float*) malloc (size_bytes);
  float *max = (float*) malloc (size_bytes);

  std::default_random_engine g (123);
  std::uniform_real_distribution<float> distr (-1.f, 1.f);
  for (int i = 0; i < size; i++) {
    min[i] = distr(g);
    max[i] = distr(g);
  }

  reference (min,
             max,
             qmin,
             qmax,
             size,
             preserve_sparsity,
             scale_ref,
             zp_ref);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  int32_t *d_zp = sycl::malloc_device<int32_t>(size, q);
  float *d_scale = sycl::malloc_device<float>(size, q);

  float *d_min = sycl::malloc_device<float>(size, q);
  q.memcpy(d_min, min, size_bytes);

  float *d_max = sycl::malloc_device<float>(size, q);
  q.memcpy(d_max, max, size_bytes);

  const int block_size = 256;
  sycl::range<1> gws ((size + block_size - 1) / block_size * block_size);
  sycl::range<1> lws (block_size);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class zp>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        zero_point(
            item,
            d_min,
            d_max,
            qmin,
            qmax,
            size,
            preserve_sparsity,
            d_scale,
            d_zp);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of zero-point kernel: %f (us)\n",
         (time * 1e-3f) / repeat);

  q.memcpy(zp, d_zp, size_bytes);
  q.memcpy(scale, d_scale, size_bytes);
  q.wait();

  bool ok = true;
  for (int i = 0; i < size; i++) {
    if (zp[i] != zp_ref[i] || scale[i] - scale_ref[i] > 1e-3f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  sycl::free(d_zp, q);
  sycl::free(d_scale, q);
  sycl::free(d_min, q);
  sycl::free(d_max, q);

  free(zp);
  free(scale);
  free(zp_ref);
  free(scale_ref);
  free(min);
  free(max);

  return 0;
}
