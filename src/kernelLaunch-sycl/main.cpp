/*
   Copyright (c) 2022 Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
 */

#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <sycl/sycl.hpp>

#define DO_NOT_OPTIMIZE_AWAY            \
  unsigned i = item.get_global_id(0);   \
    if (out) *out = args.args[i];

struct SmallKernelArgs {
  char args[16];
};

struct MediumKernelArgs {
  char args[256];
};

struct LargeKernelArgs {
  char args[4096];
};

void KernelWithSmallArgs(SmallKernelArgs args, char* out,
                         const sycl::nd_item<1> &item) { DO_NOT_OPTIMIZE_AWAY; }

void KernelWithMediumArgs(MediumKernelArgs args, char* out,
                          const sycl::nd_item<1> &item) { DO_NOT_OPTIMIZE_AWAY; }

void KernelWithLargeArgs(LargeKernelArgs args, char* out,
                         const sycl::nd_item<1> &item) { DO_NOT_OPTIMIZE_AWAY; }

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  SmallKernelArgs small_kernel_args;
  MediumKernelArgs medium_kernel_args;
  LargeKernelArgs large_kernel_args;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif
  sycl::range<1> gws (1);
  sycl::range<1> lws (1);

  auto ksFn = [=](sycl::nd_item<1> item) {
    KernelWithSmallArgs(small_kernel_args, nullptr, item);
  };

  // warmup
  for (int i = 0; i < repeat; i++) {
    q.parallel_for(sycl::nd_range<1>(gws, lws), ksFn);
  }
  q.wait();

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    q.parallel_for(sycl::nd_range<1>(gws, lws), ksFn);
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of kernelWithSmallArgs: %f (us)\n", (time * 1e-3f) / repeat);

  auto kmFn = [=](sycl::nd_item<1> item) {
    KernelWithMediumArgs(medium_kernel_args, nullptr, item);
  };

  // warmup
  for (int i = 0; i < repeat; i++) {
    q.parallel_for(sycl::nd_range<1>(gws, lws), kmFn);
  }
  q.wait();

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    q.parallel_for(sycl::nd_range<1>(gws, lws), kmFn);
  }
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of kernelWithMediumArgs: %f (us)\n", (time * 1e-3f) / repeat);

  auto klFn = [=](sycl::nd_item<1> item) {
    KernelWithLargeArgs(large_kernel_args, nullptr, item);
  };

  // warmup
  for (int i = 0; i < repeat; i++) {
    q.parallel_for(sycl::nd_range<1>(gws, lws), klFn);
  }
  q.wait();

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    q.parallel_for(sycl::nd_range<1>(gws, lws), klFn);
  }
  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of kernelWithLargeArgs: %f (us)\n", (time * 1e-3f) / repeat);

  return 0;
}
