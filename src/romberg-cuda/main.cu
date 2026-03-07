#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda.h>
#include "reference.h"

// limits of integration
#define A 0
#define B 15

// row size is related to accuracy
#define ROW_SIZE 17
#define EPS      1e-7

__host__ __device__ inline double f(double x)
{
  return exp(x)*sin(x);
}

__device__ inline unsigned int getFirstSetBitPos(int n)
{
  return log2((float)(n&-n))+1;
}

__global__ void romberg(double a, double b, double *result)  
{
  extern __shared__ double smem[];
  double diff = (b-a)/gridDim.x, step;
  int k;
  int max_eval = (1<<(ROW_SIZE-1));
  b = a + (blockIdx.x+1)*diff;
  a += blockIdx.x*diff;

  step = (b-a)/max_eval;

  double local_col[ROW_SIZE];  // specific to the row size
  for(int i = 0; i < ROW_SIZE; i++) local_col[i] = 0.0;
  if(!threadIdx.x)
  {
    k = blockDim.x;
    local_col[0] = f(a) + f(b);
  }
  else
    k = threadIdx.x;

  for(; k < max_eval; k += blockDim.x)
    local_col[ROW_SIZE - getFirstSetBitPos(k)] += 2.0*f(a + step*k);

  for(int i = 0; i < ROW_SIZE; i++)
    smem[ROW_SIZE*threadIdx.x + i] = local_col[i];
  __syncthreads();

  if(threadIdx.x < ROW_SIZE)
  {
    double sum = 0.0;
    for(int i = threadIdx.x; i < blockDim.x*ROW_SIZE; i+=ROW_SIZE)
      sum += smem[i];
    smem[threadIdx.x] = sum;
  }

  if(!threadIdx.x)
  {
    double *table = local_col;
    table[0] = smem[0];

    for(int k = 1; k < ROW_SIZE; k++)
      table[k] = table[k-1] + smem[k];

    for(int k = 0; k < ROW_SIZE; k++)  
      table[k]*= (b-a)/(1<<(k+1));

    for(int col = 0 ; col < ROW_SIZE-1 ; col++)
      for(int row = ROW_SIZE-1; row > col; row--)
        table[row] = table[row] + (table[row] - table[row-1])/((1<<(2*col+1))-1);

    result[blockIdx.x] = table[ROW_SIZE-1];
  }
}


int main( int argc, char** argv)
{
  if (argc != 4) {
    printf("Usage: %s <number of work-groups> ", argv[0]);
    printf("<work-group size> <repeat>\n");
    return 1;
  }
  const int nwg = atoi(argv[1]);
  const int wgs = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const int result_size_byte = nwg * sizeof(double);
  double *h_result = (double*) malloc (result_size_byte);

  double *d_result;
  cudaMalloc((void**) &d_result, result_size_byte);

  dim3 grids (nwg);
  dim3 blocks (wgs);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    romberg <<< grids, blocks, ROW_SIZE*wgs*sizeof(double) >>> (A,B,d_result);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", time * 1e-9f / repeat);

  // verify

  cudaMemcpy(h_result, d_result, result_size_byte, cudaMemcpyDeviceToHost);
  double sum = 0.0;
  for(int k = 0; k < nwg; k++) sum += h_result[k];

  double ref_sum = reference(f, A, B, ROW_SIZE, EPS);
  printf("%s\n", (fabs(sum - ref_sum) > EPS) ? "FAIL" : "PASS");
  if ((fabs(sum - ref_sum) > EPS)) exit(1);

  cudaFree(d_result);
  free(h_result);
  return 0;
}
