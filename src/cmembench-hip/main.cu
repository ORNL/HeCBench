/**
 * main.cpp: This file is part of the gpumembench micro-benchmark suite.
 *
 * Contact: Elias Konstantinidis <ekondis@gmail.com>
 **/

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <hip/hip_runtime.h>

#define VECTOR_SIZE 1024

// Initialize vector data
template <class T>
__device__ T init_vector(int v){
  return 0;
}

template <>
__device__ int init_vector(int v){
  return v;
}

template <>
__device__ int2 init_vector(int v){
  return make_int2(v, v);
}

template <>
__device__ int4 init_vector(int v){
  return make_int4(v, v, v, v);
}

// Sum up vector data
template <class T>
__device__ int reduce_vector(T v){
  return 0;
}

template <>
__device__ int reduce_vector(int v){
  return v;
}

template <>
__device__ int reduce_vector(int2 v){
  return v.x + v.y;
}

template <>
__device__ int reduce_vector(int4 v){
  return v.x + v.y + v.z + v.w;
}

template <class T>
__device__ void add_vector(T &target, const T &v){
}

__device__ void add_vector(int &target, const int &v) {
  target += v;
}

__device__ void add_vector(int2 &target, const int2 &v) {
  target.x += v.x;
  target.y += v.y;
}

__device__ void add_vector(int4 &target, const int4 &v) {
  target.x += v.x;
  target.y += v.y;
  target.z += v.z;
  target.w += v.w;
}

__device__ __constant__ int constant_data[VECTOR_SIZE];

template <class T>
__global__
void benchmark_constant(int *output, int repeat)
{
  T* constant_data_p = (T*)constant_data;
  T sum = init_vector<T>(0);

  for(int i=0; i<4; i++){
    for(int j=0; j<VECTOR_SIZE/(sizeof(T)/sizeof(int)); j+=4){
      add_vector(sum, constant_data_p[j+i]);
    }
  }

  if( threadIdx.x==0 && blockIdx.x==0 ) {
    *output = reduce_vector(sum);
  }
}

template<typename T>
int test_case (int *a, long gridsize, int repeat) {

  const int BLOCK_SIZE = 256;
  const int TOTAL_BLOCKS = gridsize/(BLOCK_SIZE);

  hipMemcpyToSymbol(constant_data, a, VECTOR_SIZE*sizeof(int));
  int *cd, c;
  hipMalloc((void**)&cd, sizeof(int));

  dim3 dimBlock(BLOCK_SIZE);
  dim3 dimGrid(TOTAL_BLOCKS);

  // warm up
  for (int i = 0; i < repeat; i++) { 
    benchmark_constant<T><<< dimGrid, dimBlock >>>(cd, repeat);
  }

  hipMemset(cd, 0, sizeof(int));

  hipDeviceSynchronize();

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) { 
    benchmark_constant<T><<< dimGrid, dimBlock >>>(cd, repeat);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  // verify
  hipMemcpy(&c, cd, sizeof(int), hipMemcpyDeviceToHost);
  printf("%s\t", (c == VECTOR_SIZE) ? "PASS" : "FAIL");

  hipFree(cd);

  printf("Average kernel execution time (memory access width = %zu bytes): %f ms\n",
         sizeof(T), time * 1e-6f / repeat);
  return 0;
}

void constbenchGPU(int *a, long gridsize, int repeat) {
  test_case< int>(a, gridsize, repeat);
  test_case<int2>(a, gridsize, repeat);
  test_case<int4>(a, gridsize, repeat);
}

int main(int argc, char* argv[]) {
  if (argc != 2) {
    printf("Constant memory bandwidth microbenchmark\n");
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  unsigned int datasize = VECTOR_SIZE * sizeof(int);

  int *v = (int*)malloc(datasize);
  for(int i = 0; i < VECTOR_SIZE; i++) v[i] = 1;

  constbenchGPU(v, 4096*VECTOR_SIZE, repeat);

  free(v);

  if (!(c == VECTOR_SIZE)) return 1;
  return 0;
}
