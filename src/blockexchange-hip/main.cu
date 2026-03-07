#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include "utils.h"
#include "block_load.h"
#include "block_store.h"
#include "block_exchange.h"

constexpr int items_per_thread = 4;

template <int BLOCK_SIZE, int NUM_BLOCK>
__global__ void k(const int *d, int *o, const int n)
{
  const int bid = blockIdx.x;
  const int dim = gridDim.x;
  constexpr int block_threads = BLOCK_SIZE;

  typedef BlockExchange<int, block_threads, items_per_thread> BlockExchangeT;
  typedef BlockLoad<int, block_threads, items_per_thread> LoadInteger;
  typedef BlockStore<int, block_threads, items_per_thread> StoreInteger;

  // Allocate shared memory
  __shared__ typename BlockExchangeT::TempStorage temp_storage;
  __shared__ typename LoadInteger::TempStorage loadi;
  __shared__ typename StoreInteger::TempStorage storei;

  // Obtain a segment of consecutive items that are blocked across threads
  int thread_data[items_per_thread];

  const int n_full = (NUM_BLOCK*(n/NUM_BLOCK)) + (n % NUM_BLOCK == 0 ? 0 : NUM_BLOCK);
  const int base_idx = (bid * NUM_BLOCK);
  for (unsigned int i = base_idx; i < n_full; i += dim*NUM_BLOCK) {
    unsigned int valid_items = n - i > NUM_BLOCK ? NUM_BLOCK : n - i;
    LoadInteger(loadi).Load(&(d[i]), thread_data, valid_items);

    // Collectively exchange data into a striped arrangement across threads
    BlockExchangeT(temp_storage).BlockedToStriped(thread_data, thread_data);

    StoreInteger(storei).Store(&(o[i]), thread_data, valid_items);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <number of rows> <number of columns> <repeat>\n", argv[0]);
    return 1;
  }
  const int nrows = atoi(argv[1]);
  const int ncols = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const size_t n = (size_t)nrows * ncols;
  const size_t A_size = n * sizeof(int);
  const size_t out_size = n * sizeof(int);

  int *A = (int*) malloc (A_size);
  int *out = (int*) malloc (out_size);

  for (size_t i = 0; i < n; i++) {
    A[i] = i;
  }

  int *d_A;
  hipMalloc((void**)&d_A, A_size);
  hipMemcpy(d_A, A, A_size, hipMemcpyHostToDevice);

  int *d_out;
  hipMalloc((void**)&d_out, out_size);

  const int block_size = 256;

  dim3 grid ((n+block_size-1)/block_size);
  dim3 block (block_size);

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    k<block_size, block_size*items_per_thread><<<grid, block>>>(d_A, d_out, n);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of kernel: %f (us)\n", (time * 1e-3f) / repeat);

  hipMemcpy(out, d_out, out_size, hipMemcpyDeviceToHost);

#ifdef DEBUG
  for (size_t i = 0; i < n; i++) {
    printf("%zu: %d %d\n", i, A[i], out[i]);
  }
#endif

  // TODO
  // verify the mapping between input and output

  // input:  0,1,2,3,4,5 ... 
  // output: 0,256,512,768,1,257,...
  const int items_per_block = items_per_thread * block_size;
  if (n % items_per_block == 0) {
    size_t k;
    bool ok = true;
    for (k = 0; k < n; k += items_per_block) {
      size_t i = k;
      for (int j = 0; j < items_per_thread; j++) {
        for (int m = 0; m < block_size-1; m++) {
          if (i + (m+1)*items_per_thread < n) {
            if (out[i + (m+1)*items_per_thread] - out[i + m*items_per_thread] != 1) {
              printf("Error at index %zu\n", i + (m+1)*items_per_thread);
              ok = false;
              goto stop;
            }
          }
        }
        i++;
      }
    }
    stop:
    printf("%s\n", ok ? "PASS" : "FAIL");
  }

  hipFree(d_A);
  hipFree(d_out);
  free(A);
  free(out);
  if (!ok) return 1;
  return 0;
}
