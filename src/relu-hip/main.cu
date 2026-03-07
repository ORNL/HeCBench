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
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "reference.h"

inline int divup(int count, int size) {
  return (count + size - 1)  / size;
}

// This kernel computes ReluGrad by processing one half2, two fp16, at a time.
// Kernel operation is (feature > 0) ? gradient : 0
__global__
void ReluGrad_impl1(const half*__restrict__ gradient,
                    const half*__restrict__ feature,
                          half*__restrict__ backprop,
                    const int count)
{
  int half2_count = count >> 1;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  const int total_device_threads = gridDim.x * blockDim.x;

  while (index < half2_count) {
    // The fast branch.
    // One half2, two fp16, is fetched and processed at a time.
    half2 gradient_h2 = reinterpret_cast<const half2*>(gradient)[index];
    half2 feature_h2 = reinterpret_cast<const half2*>(feature)[index];
    half2* p_backprop_h2 = reinterpret_cast<half2*>(backprop) + index;

    // Assume half2 primitives are available.
    const half2 kZero_h2 = __float2half2_rn(0.f);
    // mask = (feature > 0)
    half2 mask_h2 = __hgt2(feature_h2, kZero_h2);
    // backprop = mask * gradient
    half2 backprop_h2 = __hmul2(mask_h2, gradient_h2);

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

// This kernel computes ReluGrad by processing one half2, two fp16, at a time.
// Kernel operation is (feature > 0) ? gradient : 0
static constexpr int VectorSize = 8;

__global__
void ReluGrad_impl2(const half* __restrict__ gradient,
                    const half* __restrict__ feature,
                          half* __restrict__ backprop,
                    const int count)
{
  int half8_count = count / VectorSize;
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < half8_count) {
    float4 gradient_h8 = reinterpret_cast<const float4*>(gradient)[index];
    float4 feature_h8 = reinterpret_cast<const float4*>(feature)[index];
    float4* p_backprop_h8 = reinterpret_cast<float4*>(backprop) + index;

    half2 *gradient_h2 = reinterpret_cast<half2*>(&gradient_h8);
    half2 *feature_h2 = reinterpret_cast<half2*>(&feature_h8);
    float4 backprop_h8;
    half2* p_backprop_h2 = reinterpret_cast<half2*>(&backprop_h8);

    // Assume half2 primitives are available.
    const half2 kZero_h2 = __float2half2_rn(0.f);

    for (int i = 0; i < VectorSize / 2; i++) {
      // mask = (feature > 0)
      half2 mask_h2 = __hgt2(feature_h2[i], kZero_h2);
      // backprop = mask * gradient
      half2 backprop_h2 = __hmul2(mask_h2, gradient_h2[i]);
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

// This kernel computes Relu by processing one integer, four char, at a time.
// Kernel operation is max(input,  0)
__global__
void Relu_impl1(int count, const int* input, int* output)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    uint4 v;
    char4 b;
    b.x = input[index] & 0xFF;
    b.y = (input[index] >> 8) & 0xFF;
    b.z = (input[index] >> 16) & 0xFF;
    b.w = (input[index] >> 24) & 0xFF;
    v.x = max(b.x, 0);
    v.y = max(b.y, 0);
    v.z = max(b.z, 0);
    v.w = max(b.w, 0);
    output[index] = v.w << 24 | v.z << 16 | v.y << 8 | v.x;
  }
}

/*
// This kernel computes Relu by processing one integer, four char, at a time.
// Kernel operation is max(input,  0)
__global__
void Relu_impl2(int count, const int* input, int* output)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    output[index] = __vmaxs4(input[index], 0);
  }
}
*/

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

  h_feature = (half*) malloc (size);
  h_gradient = (half*) malloc (size);
  h_backprop = (half*) malloc (size);
  r_backprop = (half*) malloc (size);

  std::mt19937 engine(19937);
  std::uniform_real_distribution<float> real_dist(-1.f, 1.f);

  for (int i = 0; i < count; i++) {
    h_feature[i] = __float2half(real_dist(engine));
    h_gradient[i] = __float2half(1.f);
  }

  ReluGrad_reference (count, h_gradient, h_feature, r_backprop);

  half *d_gradient, *d_feature, *d_backprop;

  hipMalloc((void**)&d_gradient, size);
  hipMemcpy(d_gradient, h_gradient, size, hipMemcpyHostToDevice);

  hipMalloc((void**)&d_feature, size);
  hipMemcpy(d_feature, h_feature, size, hipMemcpyHostToDevice);

  hipMalloc((void**)&d_backprop, size);

  auto gradient_ptr = reinterpret_cast<uintptr_t>(d_gradient);
  auto feature_ptr = reinterpret_cast<uintptr_t>(d_feature);
  auto backprop_ptr = reinterpret_cast<uintptr_t>(d_backprop);

  bool aligned = gradient_ptr % 16 == 0 &&
                 feature_ptr % 16 == 0 &&
                 backprop_ptr % 16 == 0;

  printf("16-byte aligned pointers: %s\n", aligned ? "Yes" : "No");

  constexpr int kThreadInBlock = 256;
  int half_count, kBlock;

  half_count = divup(count, 2);
  kBlock = divup(half_count, kThreadInBlock);

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    ReluGrad_impl1 <<< kBlock, kThreadInBlock >>> (
        d_gradient, d_feature, d_backprop, count);

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of ReluGrad_impl1 Kernel: %f (us)\n",
          (time * 1e-3f) / repeat);

  hipMemcpy(h_backprop, d_backprop, size, hipMemcpyDeviceToHost);

  int fail = 0;
  for (int i = 0; i < count; i++) {
    if (fabsf(__half2float(h_backprop[i]) -
              __half2float(r_backprop[i])) > 1e-3f) {
      fail = 1;
      break;
    }
  }
  printf("%s\n", fail ? "FAIL" : "PASS");

  half_count = divup(count, VectorSize);
  kBlock = divup(half_count, kThreadInBlock);

  hipDeviceSynchronize();
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    ReluGrad_impl2 <<< kBlock, kThreadInBlock >>> (
        d_gradient, d_feature, d_backprop, count);

  hipDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of ReluGrad_impl2 Kernel: %f (us)\n",
          (time * 1e-3f) / repeat);

  hipMemcpy(h_backprop, d_backprop, size, hipMemcpyDeviceToHost);

  fail = 0;
  for (int i = 0; i < count; i++) {
    if (fabsf(__half2float(h_backprop[i]) -
              __half2float(r_backprop[i])) > 1e-3f) {
      fail = 1;
      break;
    }
  }
  printf("%s\n", fail ? "FAIL" : "PASS");

  //-------------------------------------------------------------------------------
  // RELU
  //-------------------------------------------------------------------------------

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
  hipMalloc((void**)&d_in, size);
  hipMemcpy(d_in, h_in, size, hipMemcpyHostToDevice);

  hipMalloc((void**)&d_out, size);
  kBlock = divup(count, kThreadInBlock);

  hipDeviceSynchronize();
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    Relu_impl1 <<< kBlock, kThreadInBlock >>> (count, d_in, d_out);

  hipDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of Relu_impl1 Kernel : %f (us)\n",
          (time * 1e-3f) / repeat);

  hipMemcpy(h_out, d_out, size, hipMemcpyDeviceToHost);

  fail = memcmp(h_out, r_out, size);
  printf("%s\n", fail ? "FAIL" : "PASS");

/*
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    Relu_impl2 <<< kBlock, kThreadInBlock >>> (count, d_in, d_out);

  hipDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of Relu_impl2 Kernel: %f (us)\n",
          (time * 1e-3f) / repeat);

  hipMemcpy(h_out, d_out, size, hipMemcpyDeviceToHost);

  fail = memcmp(h_out, r_out, size);
  printf("%s\n", fail ? "FAIL" : "PASS");
*/

  hipFree(d_out);
  hipFree(d_in);
  hipFree(d_gradient);
  hipFree(d_feature);
  hipFree(d_backprop);

  free(h_in);
  free(h_out);
  free(r_out);
  free(h_gradient);
  free(h_feature);
  free(h_backprop);
  free(r_backprop);

  if (fail) return 1;
  if (fail) return 1;
  if (fail) return 1;
  if (fail) return 1;
  return 0;
}
