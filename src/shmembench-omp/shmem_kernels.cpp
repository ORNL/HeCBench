/**
 * shmem_kernels.cu: This file is part of the gpumembench suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <chrono> // timing
#include <stdio.h>
#include <omp.h>
#include "../shmembench-cuda/reference.h"

using namespace std::chrono;

#define TOTAL_ITERATIONS (1024)
#define BLOCK_SIZE 256

typedef struct __attribute__ ((packed)) {
  float x;
  float y;
  float z;
  float w;
} float4;

// shared memory swap operation
void shmem_swap(float4 *v1, float4 *v2){
  float4 tmp;
  tmp = *v2;
  *v2 = *v1;
  *v1 = tmp;
}

// explicit conversion is required by AOMP
float4 init_val(int i){
  return {(float)i, (float)i+11, (float)i+19, (float)i+23};
}

float4 reduce_vector(float4 v1, float4 v2, float4 v3, float4 v4, float4 v5, float4 v6){
  return {v1.x + v2.x + v3.x + v4.x + v5.x + v6.x, 
          v1.y + v2.y + v3.y + v4.y + v5.y + v6.y,
          v1.z + v2.z + v3.z + v4.z + v5.z + v6.z,
          v1.w + v2.w + v3.w + v4.w + v5.w + v6.w};
}

void set_vector(float4 *target, int offset, float4 v){
  target[offset].x = v.x;
  target[offset].y = v.y;
  target[offset].z = v.z;
  target[offset].w = v.w;
}


void shmembenchGPU(float *c, const size_t size, const int n) {
  double time_shmem_128b;

  // size floats, one float4 per thread => size/4 stored values
  if (size % ((size_t)BLOCK_SIZE * 4) != 0) {
    printf("size must be a multiple of %zu\n", (size_t)BLOCK_SIZE * 4);
    printf("FAIL\n");
    return;
  }
  const size_t num_float4 = size / 4;
  int team_threads[1] = {0};

  #pragma omp target data map(from: c[0:num_float4*4]) map(tofrom: team_threads[0:1])
  {
    auto start = high_resolution_clock::now();
    for (int i = 0; i < n; i++) {
      #pragma omp target teams num_teams(num_float4/BLOCK_SIZE) thread_limit(BLOCK_SIZE)
      {
        float4 shm_buffer[BLOCK_SIZE*6];
        #pragma omp parallel num_threads(BLOCK_SIZE)
        {
          int tid = omp_get_thread_num();
          int blk = omp_get_num_threads();
          int gid = omp_get_team_num();
          int globaltid = gid * blk + tid;
          if (tid == 0 && gid == 0) team_threads[0] = blk;

          set_vector(shm_buffer, tid+0*blk, init_val(tid));
          set_vector(shm_buffer, tid+1*blk, init_val(tid+1));
          set_vector(shm_buffer, tid+2*blk, init_val(tid+3));
          set_vector(shm_buffer, tid+3*blk, init_val(tid+7));
          set_vector(shm_buffer, tid+4*blk, init_val(tid+13));
          set_vector(shm_buffer, tid+5*blk, init_val(tid+17));

          #pragma omp barrier

          #pragma unroll 32
          for(int j=0; j<TOTAL_ITERATIONS; j++){
            shmem_swap(shm_buffer+tid+0*blk, shm_buffer+tid+1*blk);
            shmem_swap(shm_buffer+tid+2*blk, shm_buffer+tid+3*blk);
            shmem_swap(shm_buffer+tid+4*blk, shm_buffer+tid+5*blk);

            #pragma omp barrier

            shmem_swap(shm_buffer+tid+1*blk, shm_buffer+tid+2*blk);
            shmem_swap(shm_buffer+tid+3*blk, shm_buffer+tid+4*blk);

            #pragma omp barrier
          }

          float4 *g_data = (float4*)c;
          g_data[globaltid] = reduce_vector(shm_buffer[tid+0*blk], 
                                            shm_buffer[tid+1*blk],
                                            shm_buffer[tid+2*blk],
                                            shm_buffer[tid+3*blk],
                                            shm_buffer[tid+4*blk],
                                            shm_buffer[tid+5*blk]);
        }
      }
    }
    auto end = high_resolution_clock::now();
    time_shmem_128b = duration_cast<nanoseconds>(end - start).count() / (double)n;
    printf("Average kernel execution time : %f (ms)\n", time_shmem_128b * 1e-6);
    // Copy results back to host memory
  }

  if (team_threads[0] != BLOCK_SIZE) {
    printf("FAIL (team threads %d, expected %d)\n", team_threads[0], BLOCK_SIZE);
    return;
  }
  int errors = shmembench_verify(c, num_float4, BLOCK_SIZE, TOTAL_ITERATIONS);
  printf("%s\n", errors ? "FAIL" : "PASS");
  if (errors) return;

  printf("Memory throughput\n");
  // 6LL keeps the product in 64-bit; 20492*size overflows 32-bit int
  const long long nops = (6LL + 4 * 5 * TOTAL_ITERATIONS + 6) * size;
  const long long operations_bytes  = nops * sizeof(float);
  const long long operations_128bit = nops / 4;

  printf("\tusing 128bit operations : %8.2f GB/sec (%6.2f billion accesses/sec)\n", 
    (double)operations_bytes / time_shmem_128b,
    (double)operations_128bit / time_shmem_128b);
}
