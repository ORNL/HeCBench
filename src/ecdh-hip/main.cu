#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <hip/hip_runtime.h>
#include "ecdh.h"

#define P_x 5
#define P_y 1
#define MODULUS 17
#define A 2

int main(int argc, char **argv)
{
  if (argc != 3) {
    printf("Usage: %s <positive number of keys> <repeat>\n", argv[0]);
    return 1;
  }
  const int num_pk = atoi(argv[1]); 
  const int repeat = atoi(argv[2]); 

  // timing fast and slow versions for making public key
  struct timeval start_fast, end_fast, start_slow, end_slow;

  const int pk_x_size = num_pk * sizeof(int);
  const int pk_y_size = num_pk * sizeof(int);

  int *pk_slow_x = (int*) malloc (pk_x_size);
  int *pk_slow_y = (int*) malloc (pk_y_size);
  int *pk_fast_x = (int*) malloc (pk_x_size);
  int *pk_fast_y = (int*) malloc (pk_y_size);

  int *d_pk_x, *d_pk_y;
  hipMalloc((void**)&d_pk_x, pk_x_size);
  hipMalloc((void**)&d_pk_y, pk_y_size);
  
  dim3 grids = (num_pk + 255) / 256;

  gettimeofday(&start_slow,NULL);

  for (int i = 0; i < repeat; i++)
    hipLaunchKernelGGL(k_slow, grids, 256, 0, 0, 18, P_x, P_y, d_pk_x, d_pk_y, MODULUS, A, num_pk);  

  hipDeviceSynchronize();
  gettimeofday(&end_slow,NULL);
  double elapsed_slow = (((end_slow.tv_sec*1000000.0 + end_slow.tv_usec) -
                        (start_slow.tv_sec*1000000.0 + start_slow.tv_usec)) / 1000000.00);

  printf("Average time (slow kernel): %f s\n", elapsed_slow / repeat);

  hipMemcpy(pk_slow_x, d_pk_x, pk_x_size, hipMemcpyDeviceToHost);
  hipMemcpy(pk_slow_y, d_pk_y, pk_y_size, hipMemcpyDeviceToHost);

  gettimeofday(&start_fast,NULL);

  for (int i = 0; i < repeat; i++)
    hipLaunchKernelGGL(k_fast, grids, 256, 0, 0, 18, P_x, P_y, d_pk_x, d_pk_y, MODULUS, A, num_pk);  

  hipDeviceSynchronize();
  gettimeofday(&end_fast,NULL);
  double elapsed_fast = (((end_fast.tv_sec*1000000.0 + end_fast.tv_usec) -
                        (start_fast.tv_sec*1000000.0 + start_fast.tv_usec)) / 1000000.00);

  printf("Average time (fast kernel): %f s\n", elapsed_fast / repeat);

  hipMemcpy(pk_fast_x, d_pk_x, pk_x_size, hipMemcpyDeviceToHost);
  hipMemcpy(pk_fast_y, d_pk_y, pk_y_size, hipMemcpyDeviceToHost);


  bool fail_pk_x = memcmp(pk_slow_x, pk_fast_x, pk_x_size);
  bool fail_pk_y = memcmp(pk_slow_y, pk_fast_y, pk_x_size);
  printf("%s\n", (fail_pk_x || fail_pk_y) ? "FAIL" : "PASS");
  
  hipFree(d_pk_x);
  hipFree(d_pk_y);
  free(pk_slow_x);
  free(pk_slow_y);
  free(pk_fast_x);
  free(pk_fast_y);
  if ((fail_pk_x || fail_pk_y)) return 1;
  return 0;
}
