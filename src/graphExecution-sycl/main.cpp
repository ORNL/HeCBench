/*
 * Copyright 1993-2018 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <stdio.h>
#include <chrono>
#include <cmath>
#include <sycl/sycl.hpp>

#define THREADS_PER_BLOCK 256
#define LAUNCH_ITERATIONS  3

namespace sycl_ext = sycl::ext::oneapi::experimental;

void reduce(const float *inputVec, double *outputVec, size_t inputSize, size_t outputSize,
            const sycl::nd_item<1> &item, double *tmp)
{
  size_t global_tid = item.get_global_id(0);
  size_t local_tid = item.get_local_id(0);
  size_t global_gid = item.get_group(0);

  double temp_sum = 0.0;
  for (size_t i = global_tid; i < inputSize;
       i += item.get_group_range(0) * item.get_local_range(0))
  {
    temp_sum += (double) inputVec[i];
  }
  tmp[local_tid] = temp_sum;

  item.barrier(sycl::access::fence_space::local_space);

  sycl::sub_group tile = item.get_sub_group();

  double beta  = temp_sum;
  double temp;

  for (int i = tile.get_local_linear_range() / 2; i > 0;
       i >>= 1) {
    if (tile.get_local_linear_id() < i) {
      temp = tmp[local_tid + i];
      beta       += temp;
      tmp[local_tid] = beta;
    }
    sycl::group_barrier(tile);
  }
  item.barrier(sycl::access::fence_space::local_space);

  if (local_tid == 0 && global_gid < outputSize) {
    beta  = 0.0;
    for (int i = 0; i < item.get_group().get_local_linear_range();
         i += tile.get_local_linear_range()) {
      beta  += tmp[i];
    }
    outputVec[global_gid] = beta;
  }
}

void reduceFinal(const double *inputVec, double *result, size_t inputSize,
                 const sycl::nd_item<1> &item, double *tmp)
{

  size_t global_tid = item.get_global_id(0);

  double temp_sum = 0.0;
  for (size_t i = global_tid; i < inputSize;
       i += item.get_group_range(0) * item.get_local_range(0))
  {
    temp_sum += inputVec[i];
  }
  tmp[item.get_local_linear_id()] = temp_sum;

  item.barrier(sycl::access::fence_space::local_space);

  //sycl::sub_group tile = item.get_sub_group();

  // do reduction in shared mem
  if ((item.get_local_range(0) >= 512) &&
      (item.get_local_linear_id() < 256))
  {
    tmp[item.get_local_linear_id()] = temp_sum =
        temp_sum + tmp[item.get_local_linear_id() + 256];
  }

  item.barrier(sycl::access::fence_space::local_space);

  if ((item.get_local_range(2) >= 256) &&
      (item.get_local_linear_id() < 128))
  {
    tmp[item.get_local_linear_id()] = temp_sum =
        temp_sum + tmp[item.get_local_linear_id() + 128];
  }

  item.barrier(sycl::access::fence_space::local_space);

  if ((item.get_local_range(2) >= 128) &&
      (item.get_local_linear_id() < 64))
  {
    tmp[item.get_local_linear_id()] = temp_sum =
        temp_sum + tmp[item.get_local_linear_id() + 64];
  }

  item.barrier(sycl::access::fence_space::local_space);

  if (item.get_local_linear_id() < 32)
  {
    // Fetch final intermediate sum from 2nd warp
    if (item.get_local_range(2) >= 64) temp_sum +=
        tmp[item.get_local_linear_id() + 32];
    // Reduce final warp using shuffle
    for (int offset = item.get_sub_group().get_local_linear_range() / 2;
         offset > 0; offset /= 2)
    {
      temp_sum +=
          sycl::shift_group_left(item.get_sub_group(), temp_sum, offset);
    }
  }
  // write result for this block to global mem
  if (item.get_local_linear_id() == 0) result[0] = temp_sum;
}

void init_input(float*a, size_t size)
{
  srand(123);
  for (size_t i=0; i < size; i++)
    a[i] = (rand() & 0xFF) / (float)RAND_MAX;
}

void usingGraph(sycl::queue &q, float* inputVec_h, float *inputVec_d,
    double *outputVec_d, double *result_d,
    size_t inputSize, size_t numOfBlocks)
{
  // compute the reference sum
  double result_r = 0.0;
  for (size_t i = 0; i < inputSize; i++)
    result_r += inputVec_h[i];

  sycl_ext::command_graph Graph{q.get_context(), q.get_device()};

  Graph.begin_recording(q);

  q.memcpy(inputVec_d, inputVec_h, sizeof(float) * inputSize);

  for (int i = 0; i < 100; i++) {
    q.memset(outputVec_d, 0, sizeof(double) * numOfBlocks);

    q.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<double, 1> tmp_acc(
          sycl::range<1>(THREADS_PER_BLOCK), cgh);

      cgh.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(numOfBlocks) *
                            sycl::range<1>(THREADS_PER_BLOCK),
                            sycl::range<1>(THREADS_PER_BLOCK)),
          [=](sycl::nd_item<1> item) {
            reduce(inputVec_d, outputVec_d, inputSize, numOfBlocks, item,
                   tmp_acc.get_multi_ptr<sycl::access::decorated::no>().get());
          });
    });

    q.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<double, 1> tmp_acc(
          sycl::range<1>(THREADS_PER_BLOCK), cgh);

      cgh.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(THREADS_PER_BLOCK),
                            sycl::range<1>(THREADS_PER_BLOCK)),
          [=](sycl::nd_item<1> item) {
            reduceFinal(outputVec_d, result_d, numOfBlocks, item,
                        tmp_acc.get_multi_ptr<sycl::access::decorated::no>().get());
          });
    });
  }

  double result_h = 0.0;
  q.memcpy(&result_h, result_d, sizeof(double));

  Graph.end_recording();

  auto ExecGraph = Graph.finalize();

  for (int i=0; i < LAUNCH_ITERATIONS; i++)
  {
    auto start = std::chrono::steady_clock::now();
    q.submit([&](sycl::handler &cgh) {
      cgh.ext_oneapi_graph(ExecGraph);
    }).wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("%s\n", (std::fabs(result_h - result_r) < 1e-6) ? "PASS" : "FAIL");
    printf("Execution time of using Graph: %f (us)\n\n", (time * 1e-3f));
  }
}

void usingStream(sycl::queue &q, float* inputVec_h, float *inputVec_d,
    double *outputVec_d, double *result_d,
    size_t inputSize, size_t numOfBlocks)
{
  // compute the reference sum
  double result_r = 0.0;
  for (size_t i = 0; i < inputSize; i++)
    result_r += inputVec_h[i];

  for (int i=0; i < LAUNCH_ITERATIONS; i++) {

    auto start = std::chrono::steady_clock::now();

    q.memcpy(inputVec_d, inputVec_h, sizeof(float) * inputSize);

    for (int i = 0; i < 100; i++) {
      q.memset(outputVec_d, 0, sizeof(double) * numOfBlocks);

      q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<double, 1> tmp_acc(
            sycl::range<1>(THREADS_PER_BLOCK), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(numOfBlocks) *
                              sycl::range<1>(THREADS_PER_BLOCK),
                              sycl::range<1>(THREADS_PER_BLOCK)),
            [=](sycl::nd_item<1> item) {
                  reduce(inputVec_d, outputVec_d, inputSize, numOfBlocks,
                         item, tmp_acc.get_multi_ptr<sycl::access::decorated::no>().get());
                });
      });
      q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<double, 1> tmp_acc(
            sycl::range<1>(THREADS_PER_BLOCK), cgh);

        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(THREADS_PER_BLOCK),
                              sycl::range<1>(THREADS_PER_BLOCK)),
            [=](sycl::nd_item<1> item) {
                  reduceFinal(outputVec_d, result_d, numOfBlocks, item,
                              tmp_acc.get_multi_ptr<sycl::access::decorated::no>().get());
                });
      });
    }

    double result_h = 0.0;
    q.memcpy(&result_h, result_d, sizeof(double)).wait();

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("%s\n", (std::fabs(result_h - result_r) < 1e-6) ? "PASS" : "FAIL");
    printf("Execution time of using Stream: %f (us)\n\n", (time * 1e-3f));
  }
}

int main(int argc, char **argv)
{
  sycl::property_list Properties{
      sycl::property::queue::in_order{},
      sycl::ext::intel::property::queue::no_immediate_command_list{}};

#ifdef USE_GPU
  sycl::queue q {sycl::gpu_selector_v, Properties};
#else
  sycl::queue q {sycl::cpu_selector_v, Properties};
#endif

  size_t maxBlocks = 512;

  for (size_t size = 512; size <= 1<<27; size = size * 512) {

    printf("\n-----------------------------\n");
    printf("%zu elements\n", size);
    printf("threads per block  = %d\n", THREADS_PER_BLOCK);
    printf("Launch iterations = %d\n", LAUNCH_ITERATIONS);

    float *inputVec_d = NULL, *inputVec_h = NULL;
    double *outputVec_d = NULL, *result_d;

    inputVec_h = (float*) malloc(sizeof(float)*size);
    inputVec_d = sycl::malloc_device<float>(size, q);
    outputVec_d = sycl::malloc_device<double>(maxBlocks, q);
    result_d = sycl::malloc_device<double>(1, q);

    init_input(inputVec_h, size);

    usingGraph(q, inputVec_h, inputVec_d, outputVec_d, result_d, size, maxBlocks);
    usingStream(q, inputVec_h, inputVec_d, outputVec_d, result_d, size, maxBlocks);

    sycl::free(inputVec_d, q);
    sycl::free(outputVec_d, q);
    sycl::free(result_d, q);
  }
  return EXIT_SUCCESS;
}
