#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "shmem_kernels.h"

#define VECTOR_SIZE (1024ULL * 1024)

int main(int argc, char* argv[]) {
  printf("Shared memory bandwidth microbenchmark\n");

  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);  // launch kernel n times

  // the kernel stores float4 values, so the buffer holds VECTOR_SIZE floats
  const size_t nbytes = VECTOR_SIZE * sizeof(float);

  printf("Buffer sizes: %zuMB\n", nbytes/(1024*1024));

  float *c = (float*)malloc(nbytes);
  memset(c, 0, nbytes);

  // benchmark execution
  shmembenchGPU(c, VECTOR_SIZE, n);

  free(c);

  return 0;
}

