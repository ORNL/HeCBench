#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda.h>
#include "reference.h"

__host__ __device__
inline float hd (const float2 ap, const float2 bp)
{
  return (ap.x - bp.x) * (ap.x - bp.x)
       + (ap.y - bp.y) * (ap.y - bp.y);
}

__device__ __forceinline__
void atomic_max(float *address, float val)
{
  unsigned int ret = __float_as_uint(*address);
  while(val > __uint_as_float(ret))
  {
    unsigned int old = ret;
    if((ret = atomicCAS((unsigned int *)address, old, __float_as_uint(val))) == old)
      break;
  }
}

__global__
void computeDistance(const float2* __restrict__ Apoints,
                     const float2* __restrict__ Bpoints,
                           float*  __restrict__ distance,
                     const int numA, const int numB)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numA) return;

  float d = std::numeric_limits<float>::max();
  float2 p = Apoints[i];
  for (int j = 0; j < numB; j++)
  {
    float t = hd(p, Bpoints[j]);
    d = std::min(t, d);
  }
  
  atomic_max(distance, d);
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <number of points in space A>", argv[0]);
    printf(" <number of points in space B> <repeat>\n");
    return 1;
  }
  const int num_Apoints = atoi(argv[1]);
  const int num_Bpoints = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const size_t num_Apoints_bytes = sizeof(float2) * num_Apoints;
  const size_t num_Bpoints_bytes = sizeof(float2) * num_Bpoints;

  float2 *h_Apoints = (float2*) malloc (num_Apoints_bytes);
  float2 *h_Bpoints = (float2*) malloc (num_Bpoints_bytes);
  
  srand(123);
  for (int i = 0; i < num_Apoints; i++) {
    h_Apoints[i].x = (float)rand() / (float)RAND_MAX;
    h_Apoints[i].y = (float)rand() / (float)RAND_MAX;
  }
  
  for (int i = 0; i < num_Bpoints; i++) {
    h_Bpoints[i].x = (float)rand() / (float)RAND_MAX;
    h_Bpoints[i].y = (float)rand() / (float)RAND_MAX;
  }

  float2 *d_Apoints, *d_Bpoints;
  float *d_distance;
  cudaMalloc((void**)&d_Apoints, num_Apoints_bytes);
  cudaMalloc((void**)&d_Bpoints, num_Bpoints_bytes);
  cudaMalloc((void**)&d_distance, 2 * sizeof(float));

  cudaMemcpy(d_Apoints, h_Apoints, num_Apoints_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Bpoints, h_Bpoints, num_Bpoints_bytes, cudaMemcpyHostToDevice);

  dim3 gridsA ((num_Apoints + 255) / 256);
  dim3 gridsB ((num_Bpoints + 255) / 256);
  dim3 blocks (256);

  float h_distance[2] = {-1.f, -1.f}; 

  double time = 0.0;

  for (int i = 0; i < repeat; i++) {
    cudaMemcpy(d_distance, h_distance, 2 * sizeof(float), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    computeDistance <<<gridsA, blocks>>> (
      d_Apoints, d_Bpoints, d_distance, num_Apoints, num_Bpoints);

    computeDistance <<<gridsB, blocks>>> (
      d_Bpoints, d_Apoints, d_distance+1, num_Bpoints, num_Apoints);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }
  printf("Average execution time of kernels: %f (ms)\n", (time * 1e-6f) / repeat);

  cudaMemcpy(h_distance, d_distance, 2 * sizeof(float), cudaMemcpyDeviceToHost);

  printf("Verifying the result may take a while..\n");
  float r_distance = hausdorff_distance(h_Apoints, h_Bpoints, num_Apoints, num_Bpoints);
  float t_distance = std::max(h_distance[0], h_distance[1]);

  bool error = (fabsf(t_distance - r_distance)) > 1e-3f;
  printf("%s\n", error ? "FAIL" : "PASS");

  free(h_Apoints);
  free(h_Bpoints);
  cudaFree(d_distance);
  cudaFree(d_Apoints);
  cudaFree(d_Bpoints);
  if (error) return 1;
  return 0;
}
