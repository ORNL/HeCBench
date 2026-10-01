#include <stdio.h>
#include <chrono>
#include <cuda.h>
#include "kernels.h"

// check the test result
void check (unsigned *err_cnt) {
  unsigned err = 0;
  // read error
  cudaMemcpy(&err, err_cnt, sizeof(unsigned), cudaMemcpyDeviceToHost);

  printf("%s", err ? "x" : ".");

  // Reset the device counter. cudaMemset is async, but cudaFree later
  // synchronizes, unlike sycl::free (https://github.com/ORNL/HeCBench/pull/344).
  cudaMemset(err_cnt, 0, sizeof(unsigned));
}

// moving inversion tests with complementary patterns
void moving_inversion (
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
  dim3 grid (1024);
  dim3 block (64);

  kernel_write <<<grid, block>>> (dev_mem, mem_size, p1);

  for(int i = 0; i < 10; i++){
    kernel_read_write <<<grid, block>>> (
        dev_mem, 
        mem_size,
        p1, p2,
        err_cnt,
        err_addr,
        err_expect,
        err_current,
        err_second_read);
    p1 = p2;
    p2 = ~p1;
  }

  kernel_read <<<grid, block>>> (dev_mem, mem_size,
      p1, 
      err_cnt,
      err_addr,
      err_expect,
      err_current,
      err_second_read);

  check(err_cnt);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  printf("Note: x indicates an error and . indicates no error when running each test\n");

  unsigned err_count = 0;

  unsigned *err_cnt;
  cudaMalloc((void**)&err_cnt, sizeof(unsigned));
  cudaMemcpy(err_cnt, &err_count, sizeof(unsigned), cudaMemcpyHostToDevice);

  unsigned long *err_addr;
  cudaMalloc((void**)&err_addr, sizeof(unsigned long) * (MAX_ERR_RECORD_COUNT));

  unsigned long *err_expect;
  cudaMalloc((void**)&err_expect, sizeof(unsigned long) * (MAX_ERR_RECORD_COUNT));

  unsigned long *err_current;
  cudaMalloc((void**)&err_current, sizeof(unsigned long) * (MAX_ERR_RECORD_COUNT));

  unsigned long *err_second_read;
  cudaMalloc((void**)&err_second_read, sizeof(unsigned long) * (MAX_ERR_RECORD_COUNT));

  // 2GB
  unsigned long mem_size = 2*1024*1024*1024UL;
  char *dev_mem;
  cudaMalloc((void**)&dev_mem, mem_size);

  printf("\ntest0: ");
  dim3 grid0 (1024);
  dim3 block0 (64);

  for (int i = 0; i < repeat; i++) {
    kernel0_write <<<grid0, block0>>> (dev_mem, mem_size);

    kernel0_read <<<grid0, block0>>> (dev_mem, mem_size,
        err_cnt,
        err_addr,
        err_expect,
        err_current,
        err_second_read);
  }

  check(err_cnt);

  printf("\ntest1: ");
  dim3 grid1 (1024);
  dim3 block1 (64);

  for (int i = 0; i < repeat; i++) {
    kernel1_write <<<grid1, block1>>> (dev_mem, mem_size);

    kernel1_read <<<grid1, block1>>> (dev_mem, mem_size,
        err_cnt,
        err_addr,
        err_expect,
        err_current,
        err_second_read);
  }

  check(err_cnt);

  printf("\ntest2: ");
  for (int i = 0; i < repeat; i++) {
    unsigned long p1 = 0;
    unsigned long p2 = ~p1;
    moving_inversion (err_cnt, err_addr, err_expect, err_current,
        err_second_read, dev_mem, mem_size, p1);

    moving_inversion (err_cnt, err_addr, err_expect, err_current,
        err_second_read, dev_mem, mem_size, p2);
  }

  printf("\ntest3: ");
  for (int i = 0; i < repeat; i++) {
    unsigned long p1 = 0x8080808080808080;
    unsigned long p2 = ~p1;
    moving_inversion (err_cnt, err_addr, err_expect, err_current,
        err_second_read, dev_mem, mem_size, p1);

    moving_inversion (err_cnt, err_addr, err_expect, err_current,
        err_second_read, dev_mem, mem_size, p2);
  }

  printf("\ntest4: ");
  srand(123);
  for (int i = 0; i < repeat; i++) {
    unsigned long p1 = rand();
    p1 = (p1 << 32) | rand();
    moving_inversion (err_cnt, err_addr, err_expect, err_current,
        err_second_read, dev_mem, mem_size, p1);
  }

  printf("\ntest5: ");
  dim3 grid5 (64*1024);
  dim3 block5 (64);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();
  
  for (int i = 0; i < repeat; i++) {
    kernel5_init <<<grid5, block5>>> (dev_mem, mem_size);

    kernel5_move <<<grid5, block5>>> (dev_mem, mem_size);

    kernel5_check <<<grid5, block5>>> (dev_mem, mem_size,
        err_cnt,
        err_addr,
        err_expect,
        err_current,
        err_second_read);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  
  check(err_cnt);

  printf("\nAverage kernel execution time (test5): %f (s)\n", (time * 1e-9f) / repeat);

  cudaFree(err_cnt);
  cudaFree(err_addr);
  cudaFree(err_expect);
  cudaFree(err_current);
  cudaFree(err_second_read);
  cudaFree(dev_mem);
  return 0;
}
