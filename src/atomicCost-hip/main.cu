#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <hip/hip_runtime.h>

static void CheckError( hipError_t err, const char *file, int line ) {
  if (err != hipSuccess) {
    printf( "%s in %s at line %d\n", hipGetErrorString( err ), file, line );
  }
}
#define CHECK_ERROR( err ) (CheckError( err, __FILE__, __LINE__ ))

#define BLOCK_SIZE 256

// measure cost of additions without atomics
template <typename T>
__global__ void woAtomicOnGlobalMem(T* result, int size)
{
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for ( unsigned int i = tid * size; i < (tid + 1) * size; i++){
    result[tid] += i % 2;
  }
}

// measure cost of additions with atomics
template <typename T>
__global__ void wiAtomicOnGlobalMem(T* result, int size)
{
  unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
  for ( unsigned int i = tid * size; i < (tid + 1) * size; i++){
    atomicAdd(&result[tid], i % 2);
  }
}

template <typename T>
void atomicCost (int length, int size, int repeat)
{
  printf("\n\n");
  printf("Each thread sums up %d elements\n", size);

  int num_threads = length / size;
  assert(length % size == 0);
  assert(num_threads % BLOCK_SIZE == 0);

  size_t result_size = sizeof(T) * num_threads;

  T* result_wi = (T*) malloc (result_size);
  T* result_wo = (T*) malloc (result_size);

  T *d_result_wi, *d_result_wo;
  CHECK_ERROR( hipMalloc((void **)&d_result_wi, result_size) );
  CHECK_ERROR( hipMemset(d_result_wi, 0, result_size) );
  CHECK_ERROR( hipMalloc((void **)&d_result_wo, result_size) );
  CHECK_ERROR( hipMemset(d_result_wo, 0, result_size) );

  dim3 block (BLOCK_SIZE);
  dim3 grid (num_threads / BLOCK_SIZE);

  CHECK_ERROR( hipDeviceSynchronize() );
  auto start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    wiAtomicOnGlobalMem<T><<<grid, block>>>(d_result_wi, size);
  }
  CHECK_ERROR( hipDeviceSynchronize() );
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of WithAtomicOnGlobalMem: %f (us)\n",
          time * 1e-3f / repeat);
  CHECK_ERROR( hipMemcpy(result_wi, d_result_wi, result_size, hipMemcpyDeviceToHost) );

  start = std::chrono::steady_clock::now();
  for(int i=0; i<repeat; i++)
  {
    woAtomicOnGlobalMem<T><<<grid, block>>>(d_result_wo, size);
  }
  CHECK_ERROR( hipDeviceSynchronize() );
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of WithoutAtomicOnGlobalMem: %f (us)\n",
          time * 1e-3f / repeat);
  CHECK_ERROR( hipMemcpy(result_wo, d_result_wo, result_size, hipMemcpyDeviceToHost) );

  int diff = memcmp(result_wi, result_wo, result_size);
  printf("%s\n", diff ? "FAIL" : "PASS");
  if (diff) exit(1);

  free(result_wi);
  free(result_wo);
  hipFree(d_result_wi);
  hipFree(d_result_wo);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <N> <repeat>\n", argv[0]);
    printf("N: the number of elements to sum per thread (1 - 16)\n");
    return 1;
  }
  const int nelems = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  const int length = 922521600;
  assert(length % BLOCK_SIZE == 0);

  printf("\nFP64 atomic add\n");
  atomicCost<double>(length, nelems, repeat);

  printf("\nINT32 atomic add\n");
  atomicCost<int>(length, nelems, repeat);

  printf("\nFP32 atomic add\n");
  atomicCost<float>(length, nelems, repeat);

  return 0;
}
