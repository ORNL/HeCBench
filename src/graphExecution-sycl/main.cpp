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
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <sycl/sycl.hpp>

#define THREADS_PER_BLOCK 256
#define INNER_ITERS 100
#define DEFAULT_LAUNCH_ITERATIONS 32

static double elapsed_us(std::chrono::steady_clock::time_point start,
                         std::chrono::steady_clock::time_point end)
{
  return std::chrono::duration<double, std::micro>(end - start).count();
}

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

static void usingStream(sycl::queue &q, float *inputVec_h, float *inputVec_d,
                     double *outputVec_d, double *result_d, double *result_h,
                     size_t inputSize, size_t numOfBlocks)
{
  q.memcpy(inputVec_d, inputVec_h, sizeof(float) * inputSize);

  for (int i = 0; i < INNER_ITERS; i++) {
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

  q.memcpy(result_h, result_d, sizeof(double));
}

static void benchmark(sycl::queue &q, float *inputVec_h, float *inputVec_d,
                      double *outputVec_d, double *result_d, size_t inputSize,
                      size_t numOfBlocks, int repeat)
{
  double result_r = 0.0;
  for (size_t i = 0; i < inputSize; i++)
    result_r += inputVec_h[i];

  double result_h = 0.0;

  // 1. Eager warmup (do not print)
  usingStream(q, inputVec_h, inputVec_d, outputVec_d, result_d, &result_h, inputSize,
           numOfBlocks);
  q.wait();

  // 2. Capture + instantiate (reported separately from replay)
  sycl_ext::command_graph Graph{q.get_context(), q.get_device()};
  auto t_cap0 = std::chrono::steady_clock::now();
  Graph.begin_recording(q);
  usingStream(q, inputVec_h, inputVec_d, outputVec_d, result_d, &result_h, inputSize,
           numOfBlocks);
  Graph.end_recording();
  auto ExecGraph = Graph.finalize();
  auto t_cap1 = std::chrono::steady_clock::now();
  printf("Capture+instantiate time: %f (us)\n", elapsed_us(t_cap0, t_cap1));

  // 3. One graph replay (do not print)
  q.submit([&](sycl::handler &cgh) { cgh.ext_oneapi_graph(ExecGraph); }).wait();

  // 4. Timed graph replays
  auto t_g0 = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    q.submit([&](sycl::handler &cgh) { cgh.ext_oneapi_graph(ExecGraph); }).wait();
  }
  auto t_g1 = std::chrono::steady_clock::now();
  printf("%s\n", (std::fabs(result_h - result_r) < 1e-6) ? "PASS" : "FAIL");
  printf("Average execution time of using Graph: %f (us)\n\n",
         elapsed_us(t_g0, t_g1) / repeat);

  // 5. Timed eager (kernels already hot)
  auto t_s0 = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    usingStream(q, inputVec_h, inputVec_d, outputVec_d, result_d, &result_h,
             inputSize, numOfBlocks);
    q.wait();
  }
  auto t_s1 = std::chrono::steady_clock::now();
  printf("%s\n", (std::fabs(result_h - result_r) < 1e-6) ? "PASS" : "FAIL");
  printf("Average execution time of using Stream: %f (us)\n\n",
         elapsed_us(t_s0, t_s1) / repeat);
}

int main(int argc, char **argv)
{
  int repeat = DEFAULT_LAUNCH_ITERATIONS;
  if (argc == 2) {
    repeat = atoi(argv[1]);
  } else if (argc > 2) {
    printf("Usage: %s [repeat]\n", argv[0]);
    return 1;
  }
  if (repeat < 1) {
    printf("repeat must be >= 1\n");
    return 1;
  }

  sycl::property_list Properties{
      sycl::property::queue::in_order{},
      sycl::ext::intel::property::queue::no_immediate_command_list{}};

#ifdef USE_GPU
  sycl::queue q {sycl::gpu_selector_v, Properties};
#else
  sycl::queue q {sycl::cpu_selector_v, Properties};
#endif

  size_t maxBlocks = 512;

  for (size_t size = 512; size <= 1 << 27; size = size * 512) {

    printf("\n-----------------------------\n");
    printf("%zu elements\n", size);
    printf("threads per block  = %d\n", THREADS_PER_BLOCK);
    printf("Launch iterations = %d\n", repeat);

    float *inputVec_d = NULL, *inputVec_h = NULL;
    double *outputVec_d = NULL, *result_d;

    inputVec_h = (float *)malloc(sizeof(float) * size);
    inputVec_d = sycl::malloc_device<float>(size, q);
    outputVec_d = sycl::malloc_device<double>(maxBlocks, q);
    result_d = sycl::malloc_device<double>(1, q);

    init_input(inputVec_h, size);

    benchmark(q, inputVec_h, inputVec_d, outputVec_d, result_d, size, maxBlocks,
              repeat);

    sycl::free(inputVec_d, q);
    sycl::free(outputVec_d, q);
    sycl::free(result_d, q);
    free(inputVec_h);
  }
  return EXIT_SUCCESS;
}
