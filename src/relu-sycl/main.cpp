/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <random>
#include <cmath>
#include <sycl/sycl.hpp>

using float4 = sycl::float4;
using uint4 = sycl::uint4;
using char4 = sycl::char4;
using half = sycl::half;
using half2 = sycl::half2;

#include "reference.h"

inline int divup(int count, int size) {
  return (count + size - 1)  / size;
}

void ReluGrad_impl1(const half *__restrict gradient,
                    const half *__restrict feature,
                          half *__restrict backprop,
                    const int count,
                    sycl::nd_item<1> &item)
{
  int half2_count = count >> 1;
  int index = item.get_global_id(0);
  const int total_device_threads =
      item.get_group_range(0) * item.get_local_range(0);

  while (index < half2_count) {
    // The fast branch.
    // One half2, two fp16, is fetched and processed at a time.
    half2 gradient_h2 =
        reinterpret_cast<const half2 *>(gradient)[index];
    half2 feature_h2 =
        reinterpret_cast<const half2 *>(feature)[index];
    half2 *p_backprop_h2 =
        reinterpret_cast<half2 *>(backprop) + index;

    const half2 kZero_h2 =
        sycl::float2{0.f, 0.f}.convert<half, sycl::rounding_mode::rte>();
    // mask = (feature > 0)
    //half2 mask_h2 = feature_h2 > kZero_h2;
    half2 mask_h2 = {feature_h2.x() > kZero_h2.x(), feature_h2.y() > kZero_h2.y()};

    // backprop = mask * gradient
    half2 backprop_h2 = mask_h2 * gradient_h2;

    // Write back the result.
    *p_backprop_h2 = backprop_h2;

    index += total_device_threads;
  }

  if ((count & 0x1) == 1 && index == half2_count) {
    // If the total number of the elements is odd, process the last element.
    half grad_h = gradient[count - 1];
    half feature_h = feature[count - 1];

    float grad_f = static_cast<float>(grad_h);
    float feature_f = static_cast<float>(feature_h);
    float backprop_f = (feature_f > 0) ? grad_f : 0;

    half backprop_h(backprop_f);
    backprop[count - 1] = backprop_h;
  }
}

static constexpr int VectorSize = 8;

void ReluGrad_impl2(const half *__restrict gradient,
                    const half *__restrict feature,
                          half *__restrict backprop,
                    const int count,
                    sycl::nd_item<1> &item)
{
  int half8_count = count / VectorSize;
  int index = item.get_global_id(0);

  if (index < half8_count) {
    float4 gradient_h8 =
        reinterpret_cast<const float4 *>(gradient)[index];
    float4 feature_h8 =
        reinterpret_cast<const float4 *>(feature)[index];
    float4 *p_backprop_h8 =
        reinterpret_cast<float4 *>(backprop) + index;

    half2 *gradient_h2 = reinterpret_cast<half2 *>(&gradient_h8);
    half2 *feature_h2 = reinterpret_cast<half2 *>(&feature_h8);
    float4 backprop_h8;
    half2 *p_backprop_h2 = reinterpret_cast<half2 *>(&backprop_h8);

    const half2 kZero_h2 =
        sycl::float2{0.f, 0.f}.convert<half, sycl::rounding_mode::rte>();

    for (int i = 0; i < VectorSize / 2; i++) {
      // mask = (feature > 0)
      half2 mask_h2 = { feature_h2[i].x() > kZero_h2.x(),
                              feature_h2[i].y() > kZero_h2.y() };

      // backprop = mask * gradient
      half2 backprop_h2 = mask_h2 * gradient_h2[i];
      p_backprop_h2[i] = backprop_h2;
    }
    // Write back the result.
    *p_backprop_h8 = backprop_h8;
  }

  int remaining_count = (count % VectorSize);

  if (index < remaining_count) {
    // Use first threads to process the remaining elements.
    half grad_h = gradient[half8_count * VectorSize + index];
    half feature_h = feature[half8_count * VectorSize + index];

    float grad_f = static_cast<float>(grad_h);
    float feature_f = static_cast<float>(feature_h);
    float backprop_f = (feature_f > 0) ? grad_f : 0;

    half backprop_h(backprop_f);
    backprop[half8_count * VectorSize + index] = backprop_h;
  }
}

void relu_impl1(int count, const int* input, int* output,
                         sycl::nd_item<1> &item)
{
  int index = item.get_global_id(0);
  if (index < count) {
    char4 b;
    char4 zero = (char4)(0);
    b.x() = input[index] & 0xFF;
    b.y() = (input[index] >> 8) & 0xFF;
    b.z() = (input[index] >> 16) & 0xFF;
    b.w() = (input[index] >> 24) & 0xFF;
    b = sycl::max(b, zero);
    output[index] = (uint)b.w() << 24 |
                    (uint)b.z() << 16 |
                    (uint)b.y() <<  8 | b.x();
  }
}

void relu_impl2(int count, const int* input, int* output,
                sycl::nd_item<1> &item)
{
  int index = item.get_global_id(0);
  if (index < count) {
    sycl::vec<int, 1> v0 {input[index]};
    sycl::vec<int, 1> v1 {0};
    auto v2 = sycl::max(v0.as<char4>(), v1.as<char4>());
    output[index] = v2.as<sycl::vec<int,1>>();
  }
}


int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <count> <repeat>\n", argv[0]);
    return 1;
  }

  const int count = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  size_t size = count * sizeof(half);

  half *h_gradient, *h_feature, *h_backprop, *r_backprop;

  h_feature = (half *)malloc(size);
  h_gradient = (half *)malloc(size);
  h_backprop = (half *)malloc(size);
  r_backprop = (half *)malloc(size);

  std::mt19937 engine(19937);
  std::uniform_real_distribution<float> real_dist(-1.f, 1.f);

  for (int i = 0; i < count; i++) {
    h_feature[i] = sycl::vec<float, 1>{real_dist(engine)}
                   .convert<half, sycl::rounding_mode::automatic>()[0];
    h_gradient[i] = sycl::vec<float, 1>{1.f}
                    .convert<half, sycl::rounding_mode::automatic>()[0];
  }

  ReluGrad_reference (count, h_gradient, h_feature, r_backprop);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  half *d_gradient, *d_feature, *d_backprop;

  d_gradient = sycl::malloc_device<half>(count, q);
  q.memcpy(d_gradient, h_gradient, size);

  d_feature = sycl::malloc_device<half>(count, q);
  q.memcpy(d_feature, h_feature, size);

  d_backprop = sycl::malloc_device<half>(count, q);

  auto gradient_ptr = reinterpret_cast<uintptr_t>(d_gradient);
  auto feature_ptr = reinterpret_cast<uintptr_t>(d_feature);
  auto backprop_ptr = reinterpret_cast<uintptr_t>(d_backprop);

  bool aligned = gradient_ptr % 16 == 0 &&
                 feature_ptr % 16 == 0 &&
                 backprop_ptr % 16 == 0;

  printf("16-byte aligned pointers: %s\n", aligned ? "Yes" : "No");

  constexpr int kThreadInBlock = 256;
  sycl::range<1> lws (kThreadInBlock);

  int half_count, kBlock;

  half_count = divup(count, 2);
  kBlock = divup(half_count, kThreadInBlock);
  sycl::range<1> gws (kBlock * kThreadInBlock);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class relu_grad>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        ReluGrad_impl1(d_gradient, d_feature, d_backprop, count, item);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of ReluGrad_impl1 Kernel: %f (us)\n",
          (time * 1e-3f) / repeat);

  q.memcpy(h_backprop, d_backprop, size).wait();

  int fail = 0;
  for (int i = 0; i < count; i++) {
    auto h = sycl::vec<half, 1>{h_backprop[i]};
    auto r = sycl::vec<half, 1>{r_backprop[i]};
    if (fabsf(h.convert<float, sycl::rounding_mode::automatic>()[0] - 
              r.convert<float, sycl::rounding_mode::automatic>()[0]) > 1e-3f) {
      fail = 1;
      break;
    }
  }
  printf("%s\n", fail ? "FAIL" : "PASS");
  if (fail) exit(1);

  half_count = divup(count, VectorSize);
  kBlock = divup(half_count, kThreadInBlock);
  gws[0] = kBlock * kThreadInBlock;

  q.wait();
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class relu_grad2>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        ReluGrad_impl2(d_gradient, d_feature, d_backprop, count, item);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of ReluGrad_impl2 Kernel: %f (us)\n",
          (time * 1e-3f) / repeat);

  q.memcpy(h_backprop, d_backprop, size).wait();

  fail = 0;
  for (int i = 0; i < count; i++) {
    auto h = sycl::vec<half, 1>{h_backprop[i]};
    auto r = sycl::vec<half, 1>{r_backprop[i]};
    if (fabsf(h.convert<float, sycl::rounding_mode::automatic>()[0] - 
              r.convert<float, sycl::rounding_mode::automatic>()[0]) > 1e-3f) {
      fail = 1;
      break;
    }
  }
  printf("%s\n", fail ? "FAIL" : "PASS");
  if (fail) exit(1);

  int *h_in, *h_out, *r_out;
  size = count * sizeof(int);
  h_in = (int*) malloc (size);
  h_out = (int*) malloc (size);
  r_out = (int*) malloc (size);  // reference

  std::uniform_int_distribution<unsigned char> int_dist(0, 255);
  for (int i = 0; i < count; i++) {
    h_in[i] = (unsigned) int_dist(engine) |
              (unsigned) int_dist(engine) << 8 |
              (unsigned) int_dist(engine) << 16 |
              (unsigned) int_dist(engine) << 24;
  }

  Relu_reference (count, h_in, r_out);

  int *d_in, *d_out;
  d_in = (int *)sycl::malloc_device(size, q);
  q.memcpy(d_in, h_in, size);

  d_out = (int *)sycl::malloc_device(size, q);

  kBlock = divup(count, kThreadInBlock);
  gws[0] = kBlock * kThreadInBlock;

  q.wait();
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class relu>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        relu_impl1(count, d_in, d_out, item);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of Relu_impl1 Kernel: %f (us)\n",
          (time * 1e-3f) / repeat);

  q.memcpy(h_out, d_out, size).wait();

  fail = memcmp(h_out, r_out, size);
  printf("%s\n", fail ? "FAIL" : "PASS");
  if (fail) exit(1);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class relu2>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        relu_impl2(count, d_in, d_out, item);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of Relu_impl2 Kernel: %f (us)\n",
          (time * 1e-3f) / repeat);

  q.memcpy(h_out, d_out, size).wait();

  fail = memcmp(h_out, r_out, size);
  printf("%s\n", fail ? "FAIL" : "PASS");
  if (fail) exit(1);

  sycl::free(d_out, q);
  sycl::free(d_in, q);
  sycl::free(d_gradient, q);
  sycl::free(d_feature, q);
  sycl::free(d_backprop, q);

  free(h_in);
  free(h_out);
  free(r_out);
  free(h_gradient);
  free(h_feature);
  free(h_backprop);
  free(r_backprop);

  return 0;
}
