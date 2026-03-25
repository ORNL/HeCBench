#include <iostream>
#include <cstdio>
#include <chrono>
#include <sycl/sycl.hpp>

#define NOW std::chrono::high_resolution_clock::now()

#include "fastdiv.h"
#include "kernels.h"

// Functional test returns 1 when it fails; otherwise it returns 0
int test(sycl::queue &q)
{
  const int blocks = 256;
  const int divisor_count = 100000;
  const int divident_count = 1000000;

  int grids = (divident_count + blocks - 1) / blocks;

  sycl::range<1> gws (grids * blocks);
  sycl::range<1> lws (blocks);

  int buf[4];
  int *d_buf = sycl::malloc_device<int>(4, q);

  std::cout << "Running functional test on " << divisor_count << " divisors, with "
            << grids * blocks << " dividents for each divisor" << std::endl;

  for(int d = 1; d < divisor_count; ++d)
  {
    for(int sign = 1; sign >= -1; sign -= 2)
    {
      int divisor = d * sign;
      q.memset(d_buf, 0, 4 * sizeof(int));

      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class test>(
          sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          check(item, divisor, d_buf);
        });
      });

      q.memcpy(buf, d_buf, 4 * sizeof(int)).wait();

      if (buf[0] > 0)
      {
        std::cout << buf[0] << " wrong results, one of them is for divident "
                  << buf[1] << ", correct quotient = " << buf[2]
                  << ", fast computed quotient = " << buf[3] << std::endl;
        sycl::free(d_buf, q);
        return 1;
      }
    }
  }

  sycl::free(d_buf, q);
  return 0;
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <repeat>\n";
    return 1;
  }
  const int repeat = atoi(argv[1]);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  const int grids = 32 * 1024;
  const int blocks = 256;

  // performance evaluation after functional test is done
  if (test(q)) return 1;

  sycl::range<1> gws (grids * blocks);
  sycl::range<1> lws (blocks);

  auto baseTpFn = [&] (sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      throughput_test<int>(item, 3, 5, 7, 0, 0);
    });
  };

  auto fastTpFn = [&] (sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      throughput_test<int_fastdiv>(item, 3, 5, 7, 0, 0);
    });
  };

  // warmup may be needed for accurate performance measurement with chrono
  for (int i = 0; i < 100; i++) {
    q.submit(baseTpFn);
    q.submit(fastTpFn);
  }
  q.wait();

  std::cout << "THROUGHPUT TEST" << std::endl;

  std::cout << "Benchmarking plain division by constant... ";
  auto start = NOW;

  for (int i = 0; i < repeat; i++) {
    q.submit(baseTpFn);
  }
  q.wait();

  auto end = NOW;
  std::chrono::duration<double> elapsed_time_slow = end-start;
  std::cout << elapsed_time_slow.count() << " seconds" << std::endl;

  std::cout << "Benchmarking fast division by constant... ";
  start = NOW;

  for (int i = 0; i < repeat; i++) {
    q.submit(fastTpFn);
  }
  q.wait();

  end = NOW;
  std::chrono::duration<double> elapsed_time_fast = end-start;
  std::cout << elapsed_time_fast.count() << " seconds" << std::endl;

  std::cout << "Speedup = " << elapsed_time_slow.count() / elapsed_time_fast.count() << std::endl;

  auto baseLatFn = [&] (sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      latency_test<int>(item, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0);
    });
  };

  auto fastLatFn = [&] (sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      latency_test<int_fastdiv>(item, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0);
    });
  };

  // warmup
  for (int i = 0; i < 100; i++) {
    q.submit(baseLatFn);
    q.submit(fastLatFn);
  }
  q.wait();

  std::cout << "LATENCY TEST" << std::endl;
  std::cout << "Benchmarking plain division by constant... ";
  start = NOW;

  for (int i = 0; i < repeat; i++) {
    q.submit(baseLatFn);
  }
  q.wait();

  end = NOW;
  elapsed_time_slow = end-start;
  std::cout << elapsed_time_slow.count() << " seconds" << std::endl;

  std::cout << "Benchmarking fast division by constant... ";
  start = NOW;

  for (int i = 0; i < repeat; i++) {
    q.submit(fastLatFn);
  }
  q.wait();

  end = NOW;
  elapsed_time_fast = end-start;
  std::cout << elapsed_time_fast.count() << " seconds" << std::endl;

  std::cout << "Speedup = " << elapsed_time_slow.count() / elapsed_time_fast.count() << std::endl;
  return 0;
}
