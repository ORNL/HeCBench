/*  Copyright (c) 2021-2022 Intel Corporation

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "reference.h"

__inline__ __device__
float exponent (float x) { return __expf(x); }

__inline__ __device__
double exponent (double x) { return exp(x); }

template <typename scalar_t, typename gscalar_t>
__global__ void loss_bwd (
    const scalar_t* __restrict__ log_softmax,
    const gscalar_t* __restrict__ grad_output,
    const gscalar_t* __restrict__ grad_output_neg,
    const int64_t* __restrict__ target,
    const scalar_t* __restrict__ weight,
    const int64_t* __restrict__ mask,
          gscalar_t* __restrict__ grad_predict)
{
  int local_id_x = threadIdx.x;
  int group_id_bs = blockIdx.y;
  int group_id_x = blockIdx.x;

  int linear_x_id = group_id_x * threadX + local_id_x;

  if (linear_x_id >= H) return;

  int offset2d = group_id_bs * H + linear_x_id;
  int idx = target[offset2d];
  int sum_offset = group_id_bs * W * H + idx * H + linear_x_id;

  gscalar_t tmp_grad;
  if (mask[offset2d])
    tmp_grad = -(grad_output[offset2d] + grad_output_neg[offset2d]);
  else
    tmp_grad = -grad_output[offset2d];

  tmp_grad = tmp_grad * weight[offset2d];

  float sum_value = tmp_grad * log_softmax[sum_offset];

  #pragma unroll
  for (int i = 0; i < W; ++i) {
    int in_offset = group_id_bs * W * H + i * H + linear_x_id;
    float tmp_sfm = exponent(log_softmax[in_offset]) * sum_value;
    float res = 0.f;
    if (i == idx) {
      res = (float)tmp_grad - tmp_sfm;
    }
    else {
      res = -tmp_sfm;
    }
    grad_predict[in_offset] = res;
  }
}

template <>
__global__ void loss_bwd<__half, __half> (
    const __half* __restrict__ log_softmax,
    const __half* __restrict__ grad_output,
    const __half* __restrict__ grad_output_neg,
    const int64_t* __restrict__ target,
    const __half* __restrict__ weight,
    const int64_t* __restrict__ mask,
          __half* __restrict__ grad_predict)
{
  int local_id_x = threadIdx.x;
  int group_id_bs = blockIdx.y;
  int group_id_x = blockIdx.x;

  int linear_x_id = group_id_x * threadX + local_id_x;

  if (linear_x_id >= H) return;

  int offset2d = group_id_bs * H + linear_x_id;
  int idx = target[offset2d];
  int sum_offset = group_id_bs * W * H + idx * H + linear_x_id;

  __half tmp_grad;
  if (mask[offset2d])
    tmp_grad = __hneg(__hadd(grad_output[offset2d] , grad_output_neg[offset2d]));
  else
    tmp_grad = __hneg(grad_output[offset2d]);

  tmp_grad = __hmul(tmp_grad , weight[offset2d]);

  float sum_value = h2f(__hmul(tmp_grad , log_softmax[sum_offset]));

  #pragma unroll
  for (int i = 0; i < W; ++i) {
    int in_offset = group_id_bs * W * H + i * H + linear_x_id;
    float tmp_sfm = h2f(hexp(log_softmax[in_offset])) * sum_value;
    float res = 0.f;
    if (i == idx) {
      res = h2f(tmp_grad) - tmp_sfm;
    }
    else {
      res = -tmp_sfm;
    }
    grad_predict[in_offset] = res;
  }
}

template <typename scalar_t, typename gscalar_t>
float loss_bwd_kernel(
  scalar_t* log_softmax,
  gscalar_t* grad_output,
  gscalar_t* grad_output_neg,
  int64_t* target, scalar_t* weight,
  int64_t* mask, gscalar_t* grad_predict)
{
  dim3 blocks ( threadX, threadBS );
  dim3 grids ( (H + threadX - 1) / threadX, bs );

  auto start = std::chrono::high_resolution_clock::now();

  hipLaunchKernelGGL(loss_bwd, grids, blocks, 0, 0, log_softmax, grad_output, grad_output_neg,
                              target, weight, mask, grad_predict);
  hipDeviceSynchronize();

  auto end = std::chrono::high_resolution_clock::now();
  float time = std::chrono::duration<float, std::milli>(end - start).count(); // ms
  return time;
}

// compute cross entropy in the backward phase
template <typename scalar_t, typename gscalar_t>
void LossNLL_BWD(int iterations) {

  vector<double> durations(3, 0.0); // timing

  const size_t grad_predict_size = sizeof(gscalar_t) * PredictShape;
  const size_t log_softmax_size = sizeof(scalar_t) * PredictShape;
  const size_t weight_size = sizeof(scalar_t) * OutputShape;
  const size_t target_size = sizeof(int64_t) * TargetShape;
  const size_t mask_size = sizeof(int64_t) * TargetShape;
  const size_t grad_output_size = sizeof(gscalar_t) * OutputShape;

  gscalar_t* grad_predict;
  hipHostMalloc((void**)&grad_predict, grad_predict_size);

  gscalar_t* grad_predict_device_host;
  hipHostMalloc((void**)&grad_predict_device_host, grad_predict_size);

  scalar_t* weight;
  hipHostMalloc((void**)&weight, weight_size);

  int64_t* target;
  hipHostMalloc((void**)&target, target_size);
  
  int64_t* mask;
  hipHostMalloc((void**)&mask, mask_size);

  scalar_t* log_softmax;
  hipHostMalloc((void**)&log_softmax, log_softmax_size);

  gscalar_t* grad_output;
  hipHostMalloc((void**)&grad_output, grad_output_size);

  gscalar_t* grad_output_neg;
  hipHostMalloc((void**)&grad_output_neg, grad_output_size);

  for (int i = 0; i < PredictShape; ++i)
    log_softmax[i] = static_cast<scalar_t>(random_float());
  for (int i = 0; i < OutputShape; ++i)
    grad_output[i] = static_cast<gscalar_t>(random_float());
  for (int i = 0; i < OutputShape; ++i)
    grad_output_neg[i] = static_cast<gscalar_t>(random_float());
  for (int i = 0; i < OutputShape; ++i)
    weight[i] = static_cast<scalar_t>(random_float());
  for (int i = 0; i < TargetShape; ++i)
    target[i] = static_cast<int64_t>(random_int(0, W - 1));
  for (int i = 0; i < TargetShape; ++i)
    mask[i] = static_cast<int64_t>(random_int(0, 1));

  // malloc device memory
  auto start = std::chrono::high_resolution_clock::now();

  gscalar_t* grad_predict_device;
  hipMalloc((void**)&grad_predict_device, grad_predict_size);

  scalar_t* weight_device;
  hipMalloc((void**)&weight_device, weight_size);

  int64_t* target_device;
  hipMalloc((void**)&target_device, target_size);
  
  int64_t* mask_device;
  hipMalloc((void**)&mask_device, mask_size);

  scalar_t* log_softmax_device;
  hipMalloc((void**)&log_softmax_device, log_softmax_size);

  gscalar_t* grad_output_device;
  hipMalloc((void**)&grad_output_device, grad_output_size);

  gscalar_t* grad_output_neg_device;
  hipMalloc((void**)&grad_output_neg_device, grad_output_size);

  hipMemcpy(grad_predict_device, grad_predict, grad_predict_size, hipMemcpyHostToDevice);
  hipMemcpy(target_device, target, target_size, hipMemcpyHostToDevice);
  hipMemcpy(mask_device, mask, mask_size, hipMemcpyHostToDevice);
  hipMemcpy(weight_device, weight, weight_size, hipMemcpyHostToDevice);
  hipMemcpy(grad_output_device, grad_output, grad_output_size, hipMemcpyHostToDevice);
  hipMemcpy(grad_output_neg_device, grad_output_neg, grad_output_size, hipMemcpyHostToDevice);
  hipMemcpy(log_softmax_device, log_softmax, log_softmax_size, hipMemcpyHostToDevice);

  auto end = std::chrono::high_resolution_clock::now();
  float time = std::chrono::duration<float, std::milli>(end - start).count(); // ms
  durations[0] += time;

  int warmup = 10;
  for (int k = 0; k < warmup + iterations; ++k) {

    float duration_gpu = loss_bwd_kernel<scalar_t, gscalar_t>(
      log_softmax_device, grad_output_device, grad_output_neg_device,
      target_device, weight_device, mask_device, grad_predict_device);

    if (k >= warmup) {
      durations[1] += duration_gpu;
    }
  }

  start = std::chrono::high_resolution_clock::now();
  hipMemcpy(grad_predict_device_host, grad_predict_device, grad_predict_size, hipMemcpyDeviceToHost);
  end = std::chrono::high_resolution_clock::now();
  time = std::chrono::duration<float, std::milli>(end - start).count(); // ms
  durations[0] += time;

  start = std::chrono::high_resolution_clock::now();

  loss_bwd_cpu<scalar_t, gscalar_t>(log_softmax, target, weight, mask, grad_output, grad_output_neg, grad_predict);

  end = std::chrono::high_resolution_clock::now();
  time = std::chrono::duration<float, std::milli>(end - start).count(); // ms
  durations[2] = time;

  verify<gscalar_t>(grad_predict, grad_predict_device_host, bs * W * H);

  std::cout << "GPU device memory allocation and data transfer time (ms) : "
            << (durations[0] / iterations) << std::endl;
  std::cout << "Average GPU kernel time (ms) : "
            << (durations[1] / iterations) << std::endl;
  std::cout << "CPU serial time (ms) : "
            << (durations[2]) << std::endl;

  double allBytes = static_cast<double>(sizeof(scalar_t)) * static_cast<double>(PredictShape * 2.0 + OutputShape * 3.0)
                  + static_cast<double>(sizeof(int64_t)) * static_cast<double>(TargetShape * 2.0);

  printf("BandWidth = %lf (GB / s)\n", allBytes / (durations[1] / iterations / 1000) / 1e9);

  hipFree(grad_predict_device);
  hipFree(weight_device);
  hipFree(target_device);
  hipFree(mask_device);
  hipFree(log_softmax_device);
  hipFree(grad_output_device);
  hipFree(grad_output_neg_device);

  hipHostFree(grad_predict);
  hipHostFree(grad_predict_device_host);
  hipHostFree(weight);
  hipHostFree(target);
  hipHostFree(mask);
  hipHostFree(log_softmax);
  hipHostFree(grad_output);
  hipHostFree(grad_output_neg);
}

int main(int argc, char** argv) {

  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  printf("Tensor size (BatchSize * Width * Height) = %d * %d * %d \n", bs, W, H);

  printf("=========== Data type is FP16 ==========\n");

  LossNLL_BWD<__half, __half>(repeat);

  printf("%s\n", (errors == 0) ? "PASS" : "FAIL");
  if (!(errors == 0)) exit(1);

  printf("=========== Data type is FP32 ==========\n");

  LossNLL_BWD<float, float>(repeat);

  printf("%s\n", (errors == 0) ? "PASS" : "FAIL");
  if (!(errors == 0)) exit(1);

  printf("=========== Data type is FP64 ==========\n");

  LossNLL_BWD<double, double>(repeat);

  printf("%s\n", (errors == 0) ? "PASS" : "FAIL");
  if (!(errors == 0)) exit(1);

  return 0;
}
