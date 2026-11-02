#include <stdio.h>
#include <chrono>
#include <hip/hip_runtime.h>
#include "kernels.h"

// check the test result
void check (unsigned *err_cnt) {
  unsigned err = 0;
  // read error
  hipMemcpy(&err, err_cnt, sizeof(unsigned), hipMemcpyDeviceToHost);

  printf("%s", err ? "x" : ".");

  // Reset the device counter. hipMemset is async, but hipFree later
  // synchronizes, unlike sycl::free (https://github.com/ORNL/HeCBench/pull/344).
  hipMemset(err_cnt, 0, sizeof(unsigned));
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

  hipLaunchKernelGGL(kernel_write, grid, block, 0, 0, dev_mem, mem_size, p1);

  for(int i = 0; i < 10; i++){
    hipLaunchKernelGGL(kernel_read_write, grid, block, 0, 0, 
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

  hipLaunchKernelGGL(kernel_read, grid, block, 0, 0, dev_mem, mem_size,
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
  hipMalloc((void**)&err_cnt, sizeof(unsigned));
  hipMemcpy(err_cnt, &err_count, sizeof(unsigned), hipMemcpyHostToDevice);

  unsigned long *err_addr;
  hipMalloc((void**)&err_addr, sizeof(unsigned long) * (MAX_ERR_RECORD_COUNT));

  unsigned long *err_expect;
  hipMalloc((void**)&err_expect, sizeof(unsigned long) * (MAX_ERR_RECORD_COUNT));

  unsigned long *err_current;
  hipMalloc((void**)&err_current, sizeof(unsigned long) * (MAX_ERR_RECORD_COUNT));

  unsigned long *err_second_read;
  hipMalloc((void**)&err_second_read, sizeof(unsigned long) * (MAX_ERR_RECORD_COUNT));

  // 2GB
  unsigned long mem_size = 2*1024*1024*1024UL;
  char *dev_mem;
  hipMalloc((void**)&dev_mem, mem_size);

  printf("\ntest0: ");
  dim3 grid0 (1024);
  dim3 block0 (64);

  for (int i = 0; i < repeat; i++) {
    hipLaunchKernelGGL(kernel0_write, grid0, block0, 0, 0, dev_mem, mem_size);

    hipLaunchKernelGGL(kernel0_read, grid0, block0, 0, 0, dev_mem, mem_size,
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
    hipLaunchKernelGGL(kernel1_write, grid1, block1, 0, 0, dev_mem, mem_size);

    hipLaunchKernelGGL(kernel1_read, grid1, block1, 0, 0, dev_mem, mem_size,
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

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();
  
  for (int i = 0; i < repeat; i++) {
    hipLaunchKernelGGL(kernel5_init, grid5, block5, 0, 0, dev_mem, mem_size);

    hipLaunchKernelGGL(kernel5_move, grid5, block5, 0, 0, dev_mem, mem_size);

    hipLaunchKernelGGL(kernel5_check, grid5, block5, 0, 0, dev_mem, mem_size,
        err_cnt,
        err_addr,
        err_expect,
        err_current,
        err_second_read);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  
  check(err_cnt);

  printf("\nAverage kernel execution time (test5): %f (s)\n", (time * 1e-9f) / repeat);

  hipFree(err_cnt);
  hipFree(err_addr);
  hipFree(err_expect);
  hipFree(err_current);
  hipFree(err_second_read);
  hipFree(dev_mem);
  return 0;
}
