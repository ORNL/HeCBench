/**
 * This file is the modified read-only mixbench GPU micro-benchmark suite.
 *
 **/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <chrono>
#include <omp.h>

#define VECTOR_SIZE (8*1024*1024)
#define granularity (8)
#define fusion_degree (4)
#define seed 0.1f

void benchmark_func(float *cd, int grid_dim, int block_dim, int compute_iterations) { 
      
  #pragma omp target teams num_teams(grid_dim) thread_limit(block_dim)
  { 
    #pragma omp parallel 
    {
      const unsigned int blockSize = block_dim;
      const int stride = blockSize;
      int idx = omp_get_team_num()*blockSize*granularity + omp_get_thread_num();
      const int big_stride = omp_get_num_teams()*blockSize*granularity;
      float tmps[granularity];
      for(int k=0; k<fusion_degree; k++) {
        #pragma unroll
        for(int j=0; j<granularity; j++) {
          // Load elements (memory intensive part)
          tmps[j] = cd[idx+j*stride+k*big_stride];

          // Perform computations (compute intensive part)
          for(int i=0; i<compute_iterations; i++)
            tmps[j] = tmps[j]*tmps[j]+(float)seed;
        }

        // Multiply add reduction
        float sum = 0;
        #pragma unroll
        for(int j=0; j<granularity; j+=2)
          sum += tmps[j]*tmps[j+1];

        #pragma unroll
        for(int j=0; j<granularity; j++)
          cd[idx+k*big_stride] = sum;
      }
    }
  }
}

void mixbenchGPU(long size, int compute_iterations, int repeat) {
  const char *benchtype = "compute with global memory (block strided)";
  printf("Trade-off type:%s\n", benchtype);
  float *cd = (float*) malloc (size*sizeof(float));
  for (int i = 0; i < size; i++) cd[i] = 0;

  const long reduced_grid_size = size/granularity/128;
  const int block_dim = 256;
  const int grid_dim = reduced_grid_size/block_dim;

  #pragma omp target data map(tofrom: cd[0:size]) 
  {
    // warmup
    for (int i = 0; i < repeat; i++) {
      benchmark_func(cd, grid_dim, block_dim, compute_iterations);
    }

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      benchmark_func(cd, grid_dim, block_dim, compute_iterations);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Total kernel execution time: %f (s)\n", time * 1e-9f);
  }

  // verification
  bool ok = true;
  for (int i = 0; i < size; i++) {
    if (cd[i] != 0) {
      if (fabsf(cd[i] - 0.050807f) > 1e-6f) {
        ok = false;
        printf("Verification failed at index %d: %f\n", i, cd[i]);
        break;
      }
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  free(cd);
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <compute iterations> <repeat>\n", argv[0]);
    return 1;
  }
  const int compute_iterations = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  unsigned int datasize = VECTOR_SIZE*sizeof(float);

  printf("Buffer size: %dMB\n", datasize/(1024*1024));

  mixbenchGPU(VECTOR_SIZE, compute_iterations, repeat);

  return 0;
}
