#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda.h>
#include "reference.h"

__global__ 
void entropy(
        float *__restrict__ d_entropy,
    const char*__restrict__ d_val, 
    int height, int width)
{
  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (y >= height || x >= width) return;

  // value of matrix element ranges from 0 inclusive to 16 exclusive
  char count[16];
  for (int i = 0; i < 16; i++) count[i] = 0;

  // total number of valid elements
  char total = 0;

  // 5x5 window
  for(int dy = -2; dy <= 2; dy++) {
    for(int dx = -2; dx <= 2; dx++) {
      int xx = x + dx;
      int yy = y + dy;
      if(xx >= 0 && yy >= 0 && yy < height && xx < width) {
        count[d_val[yy * width + xx]]++;
        total++;
      }
    }
  }

  float entropy = 0;
  if (total < 1) {
    total = 1;
  } else {
    for(int k = 0; k < 16; k++) {
      float p = __fdividef((float)count[k], (float)total);
      entropy -= p * __log2f(p);
    }
  }

  d_entropy[y * width + x] = entropy;
}

template<int bsize_x, int bsize_y>
__global__ void entropy_opt(
       float *__restrict__ d_entropy,
  const  char*__restrict__ d_val, 
  const float*__restrict__ d_logTable,
  int height, int width)
{
  __shared__ int sd_count[16][bsize_y*bsize_x];

  const int x = threadIdx.x + blockIdx.x * blockDim.x;
  const int y = threadIdx.y + blockIdx.y * blockDim.y;
  if (y >= height || x >= width) return;

  const int idx = threadIdx.y*bsize_x + threadIdx.x;

  for(int i = 0; i < 16;i++) sd_count[i][idx] = 0;

  char total = 0;
  for(int dy = -2; dy <= 2; dy++) {
    for(int dx = -2; dx <= 2; dx++) {
      int xx = x + dx,
          yy = y + dy;

      if(xx >= 0 && yy >= 0 && yy < height && xx < width) {
        sd_count[d_val[yy*width+xx]][idx]++;
        total++;
      }
    }
  }

  float entropy = 0;
  for(int k = 0; k < 16; k++)
    entropy -= d_logTable[sd_count[k][idx]];
  
  entropy = entropy / total + __log2f(total);
  d_entropy[y*width+x] = entropy;
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <width> <height> <repeat>\n", argv[0]);
    return 1;
  }
  const int width = atoi(argv[1]); 
  const int height = atoi(argv[2]); 
  const int repeat = atoi(argv[3]); 

  const int input_bytes = width * height * sizeof(char);
  const int output_bytes = width * height * sizeof(float);
  char* input = (char*) malloc (input_bytes);
  float* output = (float*) malloc (output_bytes);
  float* output_ref = (float*) malloc (output_bytes);

  srand(123);
  for (int i = 0; i < height; i++)
    for (int j = 0; j < width; j++)
      input[i * width + j] = rand() % 16;

  char* d_input;
  cudaMalloc((void**)&d_input, input_bytes);
  cudaMemcpy(d_input, input, input_bytes, cudaMemcpyHostToDevice);

  float* d_output;
  cudaMalloc((void**)&d_output, output_bytes);

  dim3 grids ((width+15)/16, (height+15)/16);
  dim3 blocks (16, 16);

  // baseline kernel
  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();
  
  for (int i = 0; i < repeat; i++)
    entropy <<< grids, blocks >>> (d_output, d_input, height, width);

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel (baseline) execution time %f (s)\n", (time * 1e-9f) / repeat);

  // optimized kernel

  float logTable[26];
  for (int i = 0; i <= 25; i++) logTable[i] = i <= 1 ? 0 : i*log2f(i);
  float* d_logTable;
  cudaMalloc((void**)&d_logTable, sizeof(logTable));
  cudaMemcpy(d_logTable, logTable, sizeof(logTable), cudaMemcpyHostToDevice);
 
  cudaDeviceSynchronize();
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    entropy_opt<16, 16> <<< grids, blocks >>> (d_output, d_input, d_logTable, height, width);

  cudaDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel (optimized) execution time %f (s)\n", (time * 1e-9f) / repeat);

  cudaMemcpy(output, d_output, output_bytes, cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_output);
  cudaFree(d_logTable);

  // verify
  reference(output_ref, input, height, width);

  bool ok = true;
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      if (fabsf(output[i * width + j] - output_ref[i * width + j]) > 1e-3f) {
        ok = false; 
        break;
      }
    }
    if (!ok) break;
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
 
  free(input);
  free(output);
  free(output_ref);
  if (!ok) return 1;
  return 0;
}
