#include <stdio.h>
#include <chrono>
#include <sycl/sycl.hpp>
#include "kernels.h"

// check the test result
void check (sycl::queue &q, unsigned *err_cnt) {
  unsigned err = 0;
  // read error
  q.memcpy(&err, err_cnt, sizeof(unsigned)).wait();

  printf("%s", err ? "x" : ".");

  // reset
  q.memset(err_cnt, 0, sizeof(unsigned)).wait();
}

// moving inversion tests with complementary patterns
void moving_inversion (
  sycl::queue &q,
  unsigned *err_cnt,
  unsigned long *err_addr,
  unsigned long *err_expect,
  unsigned long *err_current,
  unsigned long *err_second_read,
  char *dev_mem,
  unsigned long mem_size,
  unsigned long p1)
{

  unsigned long p2 = ~p1;
  sycl::range<1> gws (64*1024);
  sycl::range<1> lws (64);

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class test_write_pattern>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      kernel_write(item, dev_mem, mem_size, p1);
    });
  });

  for(int i = 0; i < 10; i++){
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class test_pattern_readwrite>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        kernel_read_write(
          item,
          dev_mem,
          mem_size,
          p1, p2,
          err_cnt,
          err_addr,
          err_expect,
          err_current,
          err_second_read);
      });
    });
    p1 = p2;
    p2 = ~p1;
  }

  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class test_pattern_read>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      kernel_read(
        item,
        dev_mem,
        mem_size,
        p1,
        err_cnt,
        err_addr,
        err_expect,
        err_current,
        err_second_read);
    });
  });

  check(q, err_cnt);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  printf("Note: x indicates an error and . indicates no error when running each test\n");

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  unsigned err_count = 0;

  unsigned *err_cnt = sycl::malloc_device<unsigned>(1, q);
  q.memcpy(err_cnt, &err_count, sizeof(unsigned));

  unsigned long *err_addr = sycl::malloc_device<unsigned long>(MAX_ERR_RECORD_COUNT, q);
  unsigned long *err_expect = sycl::malloc_device<unsigned long>(MAX_ERR_RECORD_COUNT, q);
  unsigned long *err_current = sycl::malloc_device<unsigned long>(MAX_ERR_RECORD_COUNT, q);
  unsigned long *err_second_read = sycl::malloc_device<unsigned long>(MAX_ERR_RECORD_COUNT, q);

  // 2GB
  unsigned long mem_size = 2*1024*1024*1024UL;
  char *dev_mem = sycl::malloc_device<char>(mem_size, q);

  printf("\ntest0: ");
  sycl::range<1> gws0 (64*1024);
  sycl::range<1> lws0 (64);

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class test_k0_write>(
        sycl::nd_range<1>(gws0, lws0), [=] (sycl::nd_item<1> item) {
        kernel0_write(item, dev_mem, mem_size);
      });
    });

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class test_k0_read>(
        sycl::nd_range<1>(gws0, lws0), [=] (sycl::nd_item<1> item) {
        kernel0_read(item, dev_mem, mem_size,
                     err_cnt,
                     err_addr,
                     err_expect,
                     err_current,
                     err_second_read);
      });
    });
  }

  check(q, err_cnt);

  printf("\ntest1: ");
  sycl::range<1> gws1 (64*1024);
  sycl::range<1> lws1 (64);

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class test_k1_write>(
        sycl::nd_range<1>(gws1, lws1), [=] (sycl::nd_item<1> item) {
        kernel1_write(item, dev_mem, mem_size);
      });
    });

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class test_k1_read>(
        sycl::nd_range<1>(gws1, lws1), [=] (sycl::nd_item<1> item) {
        kernel1_read(item, dev_mem, mem_size,
                     err_cnt,
                     err_addr,
                     err_expect,
                     err_current,
                     err_second_read);
      });
    });
  }
  check(q, err_cnt);

  printf("\ntest2: ");
  for (int i = 0; i < repeat; i++) {
    unsigned long p1 = 0;
    unsigned long p2 = ~p1;
    moving_inversion (q, err_cnt, err_addr, err_expect, err_current,
                      err_second_read, dev_mem, mem_size, p1);

    moving_inversion (q, err_cnt, err_addr, err_expect, err_current,
                      err_second_read, dev_mem, mem_size, p2);
  }

  printf("\ntest3: ");
  for (int i = 0; i < repeat; i++) {
    unsigned long p1 = 0x8080808080808080;
    unsigned long p2 = ~p1;
    moving_inversion (q, err_cnt, err_addr, err_expect, err_current,
                      err_second_read, dev_mem, mem_size, p1);

    moving_inversion (q, err_cnt, err_addr, err_expect, err_current,
                      err_second_read, dev_mem, mem_size, p2);
  }

  printf("\ntest4: ");
  srand(123);
  for (int i = 0; i < repeat; i++) {
    unsigned long p1 = rand();
    p1 = (p1 << 32) | rand();
    moving_inversion (q, err_cnt, err_addr, err_expect, err_current,
                      err_second_read, dev_mem, mem_size, p1);
  }

  printf("\ntest5: ");
  sycl::range<1> gws5 (64*1024);
  sycl::range<1> lws5 (64);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class test_k5_init>(
        sycl::nd_range<1>(gws5, lws5), [=] (sycl::nd_item<1> item) {
        kernel5_init(item, dev_mem, mem_size);
      });
    });

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class test_k5_move>(
        sycl::nd_range<1>(gws5, lws5), [=] (sycl::nd_item<1> item) {
        kernel5_move(item, dev_mem, mem_size);
      });
    });

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class test_k5_check>(
        sycl::nd_range<1>(gws5, lws5), [=] (sycl::nd_item<1> item) {
        kernel5_check(item, dev_mem, mem_size,
                      err_cnt,
                      err_addr,
                      err_expect,
                      err_current,
                      err_second_read);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  check(q, err_cnt);

  printf("\nAverage kernel execution time (test5): %f (s)\n", (time * 1e-9f) / repeat);

  sycl::free(err_cnt, q);
  sycl::free(err_addr, q);
  sycl::free(err_expect, q);
  sycl::free(err_current, q);
  sycl::free(err_second_read, q);
  sycl::free(dev_mem, q);
  return 0;
}
