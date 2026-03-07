#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <omp.h>

#define BLOCK_SIZE 256

// measure cost of additions without atomics
template <typename T>
void woAtomicOnGlobalMem(T* result, int size, int n)
{
  #pragma omp target teams distribute parallel for thread_limit(BLOCK_SIZE)
  for (unsigned int tid = 0; tid < n; tid++) {
    for ( unsigned int i = tid * size; i < (tid + 1) * size; i++) {
      result[tid] += i % 2;
    }
  }
}

// measure cost of additions with atomics
template <typename T>
void wiAtomicOnGlobalMem(T* result, int size, int n)
{
  #pragma omp target teams distribute parallel for thread_limit(BLOCK_SIZE)
  for (unsigned int tid = 0; tid < n; tid++) {
    for ( unsigned int i = tid * size; i < (tid + 1) * size; i++) {
      #pragma omp atomic update
      result[tid] += i % 2;
    }
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
  memset(result_wi, 0, result_size);
  memset(result_wo, 0, result_size);

  #pragma omp target data map(alloc: result_wi[0:num_threads], result_wo[0:num_threads])
  {
    auto start = std::chrono::steady_clock::now();
    for(int i=0; i<repeat; i++)
    {
      wiAtomicOnGlobalMem<T>(result_wi, size, num_threads);
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of WithAtomicOnGlobalMem: %f (us)\n",
            time * 1e-3f / repeat);
    #pragma omp target update from (result_wi[0:num_threads])

    start = std::chrono::steady_clock::now();
    for(int i=0; i<repeat; i++)
    {
      woAtomicOnGlobalMem<T>(result_wo, size, num_threads);
    }
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of WithoutAtomicOnGlobalMem: %f (us)\n",
            time * 1e-3f / repeat);
    #pragma omp target update from (result_wo[0:num_threads])

    int diff = memcmp(result_wi, result_wo, result_size);
    printf("%s\n", diff ? "FAIL" : "PASS");
    if (diff) exit(1);
  }

  free(result_wi);
  free(result_wo);
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
