#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <sycl/sycl.hpp>
#include "reference.h"

void MRCGradient (
    sycl::nd_item<1> &item,
    const int N, const int* Y, const float* X1, const float* X2, const float* dOutput,
    const float margin, float*__restrict__ dX1, float*__restrict__ dX2) {
  int i = item.get_global_id(0);
  if (i < N) {
    float dist = -Y[i] * (X1[i] - X2[i]) + margin;
    if (dist < 0.f) {
      dX1[i] = dX2[i] = 0.f;
    } else {
      dX1[i] = -Y[i] * dOutput[i];
      dX2[i] = Y[i] * dOutput[i];
    }
  }
}

void MRCGradient2(
    sycl::nd_item<1> &item,
    const int N, const int* Y, const float* X1, const float* X2, const float* dOutput,
    const float margin, float*__restrict__ dX1, float*__restrict__ dX2) {
  int i = item.get_global_id(0);
  if (i < N) {
    float y = Y[i];
    float o = dOutput[i];
    float dist = -y * (X1[i] - X2[i]) + margin;
    dX1[i] = dist < 0.f ? 0.f : -y * o;
    dX2[i] = dist < 0.f ? 0.f : y * o;
  }
}

void MRCGradient3(
    sycl::nd_item<1> &item,
    const int N, const int* Y, const float* X1, const float* X2, const float* dOutput,
    const float margin, float* __restrict__ dX1, float* __restrict__ dX2)
{
    int i = item.get_global_id(0);
    int vec_count = N / 4;

    if (i < vec_count) {
      sycl::int4   y4  = reinterpret_cast<const sycl::int4*>  (Y)      [i];
      sycl::float4 x1  = reinterpret_cast<const sycl::float4*>(X1)     [i];
      sycl::float4 x2  = reinterpret_cast<const sycl::float4*>(X2)     [i];
      sycl::float4 do4 = reinterpret_cast<const sycl::float4*>(dOutput)[i];

      sycl::float4 yf, dist;
      sycl::float4 out_x1, out_x2;
      yf.x()   = y4.x();
      yf.y()   = y4.y();
      yf.z()   = y4.z();
      yf.w()   = y4.w();
      dist.x() = sycl::fma(-yf.x() , x1.x() - x2.x() , margin);
      dist.y() = sycl::fma(-yf.y() , x1.y() - x2.y() , margin);
      dist.z() = sycl::fma(-yf.z() , x1.z() - x2.z() , margin);
      dist.w() = sycl::fma(-yf.w() , x1.w() - x2.w() , margin);

      out_x1.x() = dist.x() < 0.f ? 0.f : -yf.x() * do4.x();
      out_x2.x() = dist.x() < 0.f ? 0.f :  yf.x() * do4.x();
      out_x1.y() = dist.y() < 0.f ? 0.f : -yf.y() * do4.y();
      out_x2.y() = dist.y() < 0.f ? 0.f :  yf.y() * do4.y();
      out_x1.z() = dist.z() < 0.f ? 0.f : -yf.z() * do4.z();
      out_x2.z() = dist.z() < 0.f ? 0.f :  yf.z() * do4.z();
      out_x1.w() = dist.w() < 0.f ? 0.f : -yf.w() * do4.w();
      out_x2.w() = dist.w() < 0.f ? 0.f :  yf.w() * do4.w();

      reinterpret_cast<sycl::float4*>(dX1)[i] = out_x1;
      reinterpret_cast<sycl::float4*>(dX2)[i] = out_x2;
    }

    for (int j = 4 * vec_count + i; j < N; j += item.get_local_range(0)) {
      float yf   = (float)Y[j];
      float o    = dOutput[j];
      float dist = -yf * (X1[j] - X2[j]) + margin;
      dX1[j] = dist < 0.f ? 0.f : -yf * o;
      dX2[j] = dist < 0.f ? 0.f :  yf * o;
    }
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int length = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  size_t size_bytes = length * sizeof(float);

  float *h_X1  = (float*) malloc (size_bytes);
  float *h_X2  = (float*) malloc (size_bytes);
  float *h_O   = (float*) malloc (size_bytes);
    int *h_Y   = (  int*) malloc (size_bytes);
  float *h_dX1 = (float*) malloc (size_bytes);
  float *h_dX2 = (float*) malloc (size_bytes);
  float *r_dX1 = (float*) malloc (size_bytes);
  float *r_dX2 = (float*) malloc (size_bytes);

  const float m = 0.01;  // margin

  std::default_random_engine g (123);
  std::uniform_real_distribution<float> distr (-2.f, 2.f);
  for (int i = 0; i < length; i++) {
    h_X1[i] = distr(g);
    h_X2[i] = distr(g);
    h_O[i] = distr(g);
    h_Y[i] = (distr(g) < 0) ? -1 : 1;
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_X1, *d_X2, *d_O, *d_dX1, *d_dX2;
  int *d_Y;
  d_X1 = sycl::malloc_device<float>(length, q);
  q.memcpy(d_X1, h_X1, size_bytes);

  d_X2 = sycl::malloc_device<float>(length, q);
  q.memcpy(d_X2, h_X2, size_bytes);

  d_O = sycl::malloc_device<float>(length, q);
  q.memcpy(d_O, h_O, size_bytes);

  d_Y = sycl::malloc_device<int>(length, q);
  q.memcpy(d_Y, h_Y, size_bytes);

  d_dX1 = sycl::malloc_device<float>(length, q);
  d_dX2 = sycl::malloc_device<float>(length, q);

  int block_size = 256;
  sycl::range<1> gws ((length + block_size - 1) / block_size * block_size);
  sycl::range<1> lws (block_size);
  sycl::range<1> gws2 ((length / 4 + block_size - 1) / block_size * block_size);

  auto bwdFn = [&] (sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      MRCGradient(item, length, d_Y, d_X1, d_X2, d_O, m, d_dX1, d_dX2);
    });
  };
  auto bwd2Fn = [&] (sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      MRCGradient2(item, length, d_Y, d_X1, d_X2, d_O, m, d_dX1, d_dX2);
    });
  };
  auto bwd3Fn = [&] (sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws2, lws), [=] (sycl::nd_item<1> item) {
      MRCGradient3(item, length, d_Y, d_X1, d_X2, d_O, m, d_dX1, d_dX2);
    });
  };

  // warmup
  for (int i = 0; i < repeat; i++) {
    q.submit(bwdFn);
    q.submit(bwd2Fn);
    q.submit(bwd3Fn);
  }

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit(bwdFn);
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of MRC kernel: %f (us)\n", (time * 1e-3f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit(bwd2Fn);
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of MRC2 kernel: %f (us)\n", (time * 1e-3f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit(bwd3Fn);
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of MRC3 kernel: %f (us)\n", (time * 1e-3f) / repeat);

  // verify
  q.memcpy(h_dX1, d_dX1, size_bytes).wait();
  q.memcpy(h_dX2, d_dX2, size_bytes).wait();

  reference (length, h_Y, h_X1, h_X2, h_O, m, r_dX1, r_dX2);

  bool ok = true;
  for (int i = 0; i < length; i++) {
    if (fabs(h_dX1[i] - r_dX1[i]) > 1e-3 || fabs(h_dX2[i] - r_dX2[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  sycl::free(d_X1, q);
  sycl::free(d_X2, q);
  sycl::free(d_O, q);
  sycl::free(d_Y, q);
  sycl::free(d_dX1, q);
  sycl::free(d_dX2, q);

  free(h_X1);
  free(h_X2);
  free(h_O);
  free(h_Y);
  free(h_dX1);
  free(h_dX2);

  return 0;
}
