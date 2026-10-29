/*
   Shared memory speeds up performance when we need to access data frequently. 
   Here, the 1D stencil kernel adds all its neighboring data within a radius.

   The C model is added to verify the stencil result on a GPU
*/

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <omp.h>

#define RADIUS 7
#define BLOCK_SIZE 256

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <length> <repeat>\n", argv[0]);
    printf("length is a multiple of %d\n", BLOCK_SIZE);
    return 1;
  }
  const int length = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
  const int warmup = 100;

  int size = length;
  int pad_size = (length + RADIUS);

  // Alloc space for host copies of a, b, c and setup input values
  int* a = (int *)malloc(pad_size*sizeof(int)); 
  int* b = (int *)malloc(size*sizeof(int));

  for (int i = 0; i < length+RADIUS; i++) a[i] = i;

  const int num_blocks = length / BLOCK_SIZE;

  std::chrono::steady_clock::time_point start;

  #pragma omp target data map(to: a[0:pad_size]) map(from:b[0:size])
  {
    // Warmup launches load the device kernel. Keep them outside the timer.
    for (int iter = 0; iter < repeat + warmup; iter++) {
      if (iter == warmup)
        start = std::chrono::steady_clock::now();

      #pragma omp target teams distribute
      for (int block = 0; block < num_blocks; ++block) {
        const int i = block * BLOCK_SIZE;
        int temp[BLOCK_SIZE + 2 * RADIUS];
        #pragma omp parallel for schedule(static,1)
        for (int j = 0; j < BLOCK_SIZE; j++) {
          int gindex = i+j;
          temp[j+RADIUS] = a[gindex]; 
          if (j < RADIUS) {
            temp[j] = (gindex < RADIUS) ? 0 : a[gindex - RADIUS];
            temp[j + RADIUS + BLOCK_SIZE] = a[gindex + BLOCK_SIZE];
          }
        }

        #pragma omp parallel for schedule(static,1)
        for (int j = 0; j < BLOCK_SIZE; j++) {
          int result = 0;
          for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
            result += temp[j+RADIUS+offset];
          b[i+j] = result; 
        }
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);
  }

  // verification
  bool ok = true;
  for (int i = 0; i < 2*RADIUS; i++) {
    int s = 0;
    for (int j = i; j <= i+2*RADIUS; j++)
      s += j < RADIUS ? 0 : (a[j] - RADIUS);
    if (s != b[i]) {
      printf("Error at %d: %d (host) != %d (device)\n", i, s, b[i]);
      ok = false;
      break;
    }
  }

  for (int i = 2*RADIUS; i < length; i++) {
    int s = 0;
    for (int j = i-RADIUS; j <= i+RADIUS; j++)
      s += a[j];
    if (s != b[i]) {
      printf("Error at %d: %d (host) != %d (device)\n", i, s, b[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  // Cleanup
  free(a);
  free(b); 
  return 0;
}
