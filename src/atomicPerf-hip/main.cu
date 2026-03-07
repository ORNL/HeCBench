#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <hip/hip_runtime.h>

#define BLOCK_SIZE 256

#include "reference.h"

static void CheckError( hipError_t err, const char *file, int line ) {
  if (err != hipSuccess) {
    printf( "%s in %s at line %d\n", hipGetErrorString( err ), file, line );
  }
}
#define CHECK_ERROR( err ) (CheckError( err, __FILE__, __LINE__ ))


template <typename T>
__global__ void BlockRangeAtomicOnGlobalMem(T* data, int n)
{
  unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  for ( unsigned int i = tid; i < n; i += blockDim.x*gridDim.x){
    atomicAdd(data+threadIdx.x, (T)1);  //arbitrary number to add
  }
}

template <typename T>
__global__ void WarpRangeAtomicOnGlobalMem(T* data, int n)
{
  unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  for ( unsigned int i = tid; i < n; i += blockDim.x*gridDim.x){
    atomicAdd(data+(i & 0x1F), (T)1); //arbitrary number to add
  }
}

template <typename T>
__global__ void SingleRangeAtomicOnGlobalMem(T* data, int offset, int n)
{
  unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  for ( unsigned int i = tid; i < n; i += blockDim.x*gridDim.x){
    atomicAdd(data+offset, (T)1);    //arbitrary number to add
  }
}

template <typename T>
__global__ void BlockRangeAtomicOnSharedMem(T* data, int n)
{
  __shared__ T smem_data[BLOCK_SIZE];
  unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  for ( unsigned int i = tid; i < n; i += blockDim.x*gridDim.x){
    atomicAdd(smem_data+threadIdx.x, (T)1);
  }
  if (blockIdx.x == gridDim.x)
    data[threadIdx.x] = smem_data[threadIdx.x];
}

template <typename T>
__global__ void WarpRangeAtomicOnSharedMem(T* data, int n)
{
  __shared__ T smem_data[32];
  unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  for ( unsigned int i = tid; i < n; i += blockDim.x*gridDim.x){
    atomicAdd(smem_data+(i & 0x1F), (T)1);
  }
  if (blockIdx.x == gridDim.x && threadIdx.x < 0x1F)
    data[threadIdx.x] = smem_data[threadIdx.x];
}

template <typename T>
__global__ void SingleRangeAtomicOnSharedMem(T* data, int offset, int n)
{
  __shared__ T smem_data[BLOCK_SIZE];
  unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
  for ( unsigned int i = tid; i < n; i += blockDim.x*gridDim.x){
    atomicAdd(smem_data + offset, (T)1);
  }
  if (blockIdx.x == gridDim.x && threadIdx.x == 0)
    data[threadIdx.x] = smem_data[threadIdx.x];
}

template <typename T>
void atomicPerf (int n, int t, int repeat)
{
  size_t data_size = sizeof(T) * t;

  T* data = (T*) malloc (data_size);
  T* h_data = (T*) malloc (data_size);
  T* r_data = (T*) malloc (data_size);
  int fail;

  for(int i=0; i<t; i++) {
    data[i] = i%1024+1;
  }

  T* d_data;
  CHECK_ERROR( hipMalloc((void **)&d_data, data_size) );

  dim3 block (BLOCK_SIZE);
  dim3 grid (n / BLOCK_SIZE);

  CHECK_ERROR( hipMemcpy(d_data, data, data_size, hipMemcpyHostToDevice) );
  CHECK_ERROR( hipDeviceSynchronize() );
  auto start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    BlockRangeAtomicOnGlobalMem<T><<<grid, block>>>(d_data, n);
  }
  CHECK_ERROR( hipDeviceSynchronize() );
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of BlockRangeAtomicOnGlobalMem: %f (us)\n",
          time * 1e-3f / repeat);

  CHECK_ERROR( hipMemcpy(h_data, d_data, data_size, hipMemcpyDeviceToHost) );
  memcpy(r_data, data, data_size);
  for(int i=0; i<repeat; i++)
    BlockRangeAtomicOnGlobalMem_ref<T>(r_data, n);
  fail = memcmp(h_data, r_data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");
  
  CHECK_ERROR( hipMemcpy(d_data, data, data_size, hipMemcpyHostToDevice) );
  CHECK_ERROR( hipDeviceSynchronize() );
  start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    WarpRangeAtomicOnGlobalMem<T><<<grid, block>>>(d_data, n);
  }
  CHECK_ERROR( hipDeviceSynchronize() );
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of WarpRangeAtomicOnGlobalMem: %f (us)\n",
          time * 1e-3f / repeat);

  CHECK_ERROR( hipMemcpy(h_data, d_data, data_size, hipMemcpyDeviceToHost) );
  memcpy(r_data, data, data_size);
  for(int i=0; i<repeat; i++)
    WarpRangeAtomicOnGlobalMem_ref<T>(r_data, n);
  fail = memcmp(h_data, r_data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");

  CHECK_ERROR( hipMemcpy(d_data, data, data_size, hipMemcpyHostToDevice) );
  CHECK_ERROR( hipDeviceSynchronize() );
  start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    SingleRangeAtomicOnGlobalMem<T><<<grid, block>>>(d_data, i % BLOCK_SIZE, n);
  }
  CHECK_ERROR( hipDeviceSynchronize() );
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of SingleRangeAtomicOnGlobalMem: %f (us)\n",
          time * 1e-3f / repeat);

  CHECK_ERROR( hipMemcpy(h_data, d_data, data_size, hipMemcpyDeviceToHost) );
  memcpy(r_data, data, data_size);
  for(int i=0; i<repeat; i++)
    SingleRangeAtomicOnGlobalMem_ref<T>(r_data, i % BLOCK_SIZE, n);
  fail = memcmp(h_data, r_data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");

  CHECK_ERROR( hipMemcpy(d_data, data, data_size, hipMemcpyHostToDevice) );
  CHECK_ERROR( hipDeviceSynchronize() );
  start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    BlockRangeAtomicOnSharedMem<T><<<grid, block>>>(d_data, n);
  }
  CHECK_ERROR( hipDeviceSynchronize() );
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of BlockRangeAtomicOnSharedMem: %f (us)\n",
          time * 1e-3f / repeat);

  CHECK_ERROR( hipMemcpy(h_data, d_data, data_size, hipMemcpyDeviceToHost) );
  fail = memcmp(h_data, data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");

  CHECK_ERROR( hipMemcpy(d_data, data, data_size, hipMemcpyHostToDevice) );
  CHECK_ERROR( hipDeviceSynchronize() );
  start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    WarpRangeAtomicOnSharedMem<T><<<grid, block>>>(d_data, n);
  }
  CHECK_ERROR( hipDeviceSynchronize() );
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of WarpRangeAtomicOnSharedMem: %f (us)\n",
          time * 1e-3f / repeat);

  CHECK_ERROR( hipMemcpy(h_data, d_data, data_size, hipMemcpyDeviceToHost) );
  fail = memcmp(h_data, data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");

  CHECK_ERROR( hipMemcpy(d_data, data, data_size, hipMemcpyHostToDevice) );
  CHECK_ERROR( hipDeviceSynchronize() );
  start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    SingleRangeAtomicOnSharedMem<T><<<grid, block>>>(d_data, i % BLOCK_SIZE, n);
  }
  CHECK_ERROR( hipDeviceSynchronize() );
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of SingleRangeAtomicOnSharedMem: %f (us)\n",
          time * 1e-3f / repeat);

  CHECK_ERROR( hipMemcpy(h_data, d_data, data_size, hipMemcpyDeviceToHost) );
  fail = memcmp(h_data, data, data_size);
  printf("%s\n", fail ? "FAIL" : "PASS");

  free(data);
  free(h_data);
  free(r_data);
  hipFree(d_data); 
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const int n = 3*4*7*8*9*256; // number of threads
  const int len = 1024; // data array length
  
  printf("\nFP64 atomic add\n");
  atomicPerf<double>(n, len, repeat); 

  printf("\nINT32 atomic add\n");
  atomicPerf<int>(n, len, repeat); 

  printf("\nFP32 atomic add\n");
  atomicPerf<float>(n, len, repeat); 

  if (fail) return 1;
  if (fail) return 1;
  if (fail) return 1;
  if (fail) return 1;
  if (fail) return 1;
  if (fail) return 1;
  return 0;
}
