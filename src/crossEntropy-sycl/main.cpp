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
#include <sycl/sycl.hpp>

using namespace std;

// generate rand int64_t [a, b]
#define random_int(a, b) ( rand() % (b - a) + a )

// generate rand float [0, 1]
#define random_float() (rand() / double(RAND_MAX))

// This version fused the log_softmax
#define tolerance 4e-3

// tunable thread block size
#define threadX  64
#define threadBS 1

int64_t errors(0);

constexpr int bs = 128;
constexpr int W = 81;
constexpr int H = 8732;
constexpr int PredictShape = bs * W * H;
constexpr int TargetShape = bs * H;
constexpr int OutputShape = bs * H;

template <typename scalar_t, typename gscalar_t>
float loss_bwd_kernel(
  sycl::queue& q,
  scalar_t* log_softmax,
  gscalar_t* grad_output,
  gscalar_t* grad_output_neg,
  int64_t* target, scalar_t* weight,
  int64_t* mask, gscalar_t* grad_predict)
{
  sycl::range<2> lws ( threadBS, threadX );
  sycl::range<2> gws ( bs, (H + threadX - 1) / threadX * threadX );

  auto start = std::chrono::high_resolution_clock::now();

  q.submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<2> (gws, lws), [=](sycl::nd_item<2> item) { 
      int local_id_x = item.get_local_id(1);
      int group_id_bs = item.get_group(0);
      int group_id_x = item.get_group(1);
      
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
        float tmp_sfm = sycl::exp(log_softmax[in_offset]) * sum_value;
        float res = 0.0;
        if (i == idx) {
          res = tmp_grad - tmp_sfm;
        } else {
          res = -tmp_sfm;
        }
        grad_predict[in_offset] = res;
      }
    });
  }).wait();

  auto end = std::chrono::high_resolution_clock::now();
  float time = std::chrono::duration<float, std::milli>(end - start).count(); // ms
  return time;
}

template <typename scalar_t, typename gscalar_t>
void loss_bwd_cpu(scalar_t* predict, int64_t* target, scalar_t* weight, int64_t* mask,
                  gscalar_t* grad_output, gscalar_t* grad_output_neg, gscalar_t* grad_predict)
{
  for (int i = 0; i < bs; ++i) {
    for (int k = 0; k < W; ++k) {
      for (int j = 0; j < H; ++j) {
        int64_t offset = i * H + j;
        int64_t idx = target[offset];
        int64_t predict_offset = i * W * H + k * H + j;

        if (idx == int64_t(k)) {
          grad_predict[predict_offset] = (-grad_output[offset] * weight[offset]) + 
                                         (-grad_output_neg[offset] * weight[offset] * mask[offset]);
        }
        else {
          grad_predict[predict_offset] = 0;
        }
      }
    }
  }

  vector<vector<float>> sum_value;
  for (int i = 0; i < bs; ++i) {
    vector<float> bs_sum_value;
    for (int j = 0; j < H; ++j) {
      float sum = 0.0;
      for (int k = 0; k < W; ++k) {
        int64_t offset = i * W * H + k * H + j;
        sum += grad_predict[offset] * predict[offset];
      }
      bs_sum_value.push_back(sum);
    }
    sum_value.push_back(bs_sum_value);
  }

  for (int i = 0; i < bs; ++i) {
    for (int k = 0; k < W; ++k) {
      for (int j = 0; j < H; ++j) {
        int64_t offset = i * W * H + k * H + j;
        grad_predict[offset] = grad_predict[offset] - std::exp(predict[offset]) * sum_value[i][j];
      }
    }
  }
}

template <typename scalar_t>
void verify(scalar_t* output, scalar_t* output_device, size_t sz) {
  int count = 0;
  for (size_t i = 0; i < sz; ++i) {
    int64_t offset = i;
    if (std::abs(output[offset] - output_device[offset]) > tolerance) {
      count++;
      if (count < 10) 
        std::cout << "Error, output not equal, i="
                  << i << ", cpu_result = " << output[offset] << ", device_result = "
                  << output_device[offset] << ", gap = "
                  << (output[offset] - output_device[offset]) << std::endl;
    }
  }
  errors += count;
}

// compute cross entropy in the backward phase
template <typename scalar_t, typename gscalar_t>
void LossNLL_BWD(sycl::queue& q, int iterations) {

  vector<double> durations(3, 0.0); // timing

  gscalar_t* grad_predict = sycl::malloc_host<gscalar_t>(PredictShape, q);
  scalar_t* log_softmax = sycl::malloc_host<scalar_t>(PredictShape, q);
  gscalar_t* grad_output = sycl::malloc_host<gscalar_t>(OutputShape, q);
  gscalar_t* grad_output_neg = sycl::malloc_host<gscalar_t>(OutputShape, q);
  int64_t* target = sycl::malloc_host<int64_t>(TargetShape, q);
  int64_t* mask = sycl::malloc_host<int64_t>(TargetShape, q);
  scalar_t* weight = sycl::malloc_host<scalar_t>(OutputShape, q);
  gscalar_t* grad_predict_device_host = sycl::malloc_host<gscalar_t>(PredictShape, q);

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

  gscalar_t* grad_predict_device = sycl::malloc_device<gscalar_t>(PredictShape, q);
  scalar_t* log_softmax_device = sycl::malloc_device<scalar_t>(PredictShape, q);
  gscalar_t* grad_output_device = sycl::malloc_device<gscalar_t>(OutputShape, q);
  gscalar_t* grad_output_neg_device = sycl::malloc_device<gscalar_t>(OutputShape, q);
  int64_t* target_device = sycl::malloc_device<int64_t>(TargetShape, q);
  int64_t* mask_device = sycl::malloc_device<int64_t>(TargetShape, q);
  scalar_t* weight_device = sycl::malloc_device<scalar_t>(OutputShape, q);

  q.memcpy(log_softmax_device, log_softmax, PredictShape * sizeof(scalar_t));
  q.memcpy(grad_output_device, grad_output, OutputShape * sizeof(gscalar_t));
  q.memcpy(grad_output_neg_device, grad_output_neg, OutputShape * sizeof(gscalar_t));
  q.memcpy(weight_device, weight, OutputShape * sizeof(scalar_t));
  q.memcpy(target_device, target, TargetShape * sizeof(int64_t));
  q.memcpy(mask_device, mask, TargetShape * sizeof(int64_t));
  q.wait();

  auto end = std::chrono::high_resolution_clock::now();
  float time = std::chrono::duration<float, std::milli>(end - start).count(); // ms
  durations[0] += time;

  int warmup = 10;
  for (int k = 0; k < warmup + iterations; ++k) {

    float duration_gpu = loss_bwd_kernel<scalar_t, gscalar_t>(
      q, log_softmax_device, grad_output_device, grad_output_neg_device,
      target_device, weight_device, mask_device, grad_predict_device);

    if (k >= warmup) {
      durations[1] += duration_gpu;
    }
  }

  start = std::chrono::high_resolution_clock::now();
  q.memcpy(grad_predict_device_host, grad_predict_device, PredictShape * sizeof(gscalar_t)).wait();
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

  sycl::free(grad_predict_device, q);
  sycl::free(weight_device, q);
  sycl::free(target_device, q);
  sycl::free(mask_device, q);
  sycl::free(log_softmax_device, q);
  sycl::free(grad_output_device, q);
  sycl::free(grad_output_neg_device, q);

  sycl::free(grad_predict, q);
  sycl::free(grad_predict_device_host, q);
  sycl::free(weight, q);
  sycl::free(target, q);
  sycl::free(mask, q);
  sycl::free(log_softmax, q);
  sycl::free(grad_output, q);
  sycl::free(grad_output_neg, q);
}

int main(int argc, char** argv) {

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

  printf("Tensor size (BatchSize * Width * Height) = %d * %d * %d \n", bs, W, H);

  printf("=========== Data type is FP16 ==========\n");

  LossNLL_BWD<sycl::half, sycl::half>(q, repeat);

  printf("%s\n", (errors == 0) ? "PASS" : "FAIL");
  if (!(errors == 0)) exit(1);

  printf("=========== Data type is FP32 ==========\n");

  LossNLL_BWD<float, float>(q, repeat);

  printf("%s\n", (errors == 0) ? "PASS" : "FAIL");
  if (!(errors == 0)) exit(1);

  printf("=========== Data type is FP64 ==========\n");

  LossNLL_BWD<double, double>(q, repeat);

  printf("%s\n", (errors == 0) ? "PASS" : "FAIL");
  if (!(errors == 0)) exit(1);

  return 0;
}
