#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cuda.h>
#include "block_load.h"
#include "block_store.h"

#define GPU_CHECK(x) do { \
  cudaError_t err = x; \
  if (err != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(1); \
  } \
} while (0)

__global__ void reference (const float * __restrict__ A,
                           unsigned char *out, const size_t n)
{
  for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
       idx < n/4; idx += gridDim.x * blockDim.x) {
    const float4 v = reinterpret_cast<const float4*>(A)[idx];
    uchar4 o;
    o.x = (int)v.x;
    o.y = (int)v.y;
    o.z = (int)v.z;
    o.w = (int)v.w;
    reinterpret_cast<uchar4*>(out)[idx] = o;
  }
}

template<int BLOCKSIZE, int ITEMS_PER_THREAD>
__global__ void kernel (const float * __restrict__ A,
                        unsigned char *out, const size_t n)
{
  float vals[ITEMS_PER_THREAD];
  unsigned char qvals[ITEMS_PER_THREAD];

  typedef BlockLoad<float, BLOCKSIZE, ITEMS_PER_THREAD> LoadFloat;
  typedef BlockStore<unsigned char, BLOCKSIZE, ITEMS_PER_THREAD> StoreChar;

  __shared__ typename LoadFloat::TempStorage loadf_storage;
  __shared__ typename StoreChar::TempStorage storec_storage;

  for (size_t i = (size_t)blockIdx.x * BLOCKSIZE * ITEMS_PER_THREAD;
       i < n; i += gridDim.x * BLOCKSIZE * ITEMS_PER_THREAD)
  {
      int valid_items = min(n - i, (size_t)BLOCKSIZE * ITEMS_PER_THREAD);

      // Parameters:
      // block_src_it – [in] The thread block's base iterator for loading from
      // dst_items – [out] Destination to load data into
      // block_items_end – [in] Number of valid items to load
      LoadFloat(loadf_storage).Load(&(A[i]), vals, valid_items);

      #pragma unroll
      for(int j = 0; j < ITEMS_PER_THREAD; j++)
          qvals[j] = int(vals[j]);

      StoreChar(storec_storage).Store(&(out[i]), qvals, valid_items);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Block access N elements where N is represented as rows x columns\n");
    printf("Usage: %s <number of rows> <number of columns> <repeat>\n", argv[0]);
    return 1;
  }
  const int nrows = atoi(argv[1]);
  const int ncols = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const size_t n = ((size_t)nrows * ncols + 3) / 4 * 4;
  const size_t A_size = n * sizeof(float);
  const size_t out_size = n * sizeof(unsigned char);

  float *A = (float*) malloc (A_size);
  unsigned char *out = (unsigned char*) malloc (out_size);
  unsigned char *out_ref = (unsigned char*) malloc (out_size);

  std::mt19937 gen{19937};

  std::normal_distribution<float> d{-128.0, 127.0};

  for (size_t i = 0; i < n; i++) {
    A[i] = d(gen);
  }

  float *d_A;
  GPU_CHECK(cudaMalloc((void**)&d_A, A_size));
  GPU_CHECK(cudaMemcpy(d_A, A, A_size, cudaMemcpyHostToDevice));

  unsigned char *d_out, *d_out_ref;
  GPU_CHECK(cudaMalloc((void**)&d_out, out_size));
  GPU_CHECK(cudaMalloc((void**)&d_out_ref, out_size));

  cudaDeviceProp prop;
  GPU_CHECK(cudaGetDeviceProperties(&prop, 0));
  dim3 grid (16 * prop.multiProcessorCount);
  const int items_per_thread = 4;
  const int block_size = 256;
  dim3 block (block_size);

  reference<<<grid, block>>>(d_A, d_out_ref, n);
  kernel<block_size, items_per_thread><<<grid, block>>>(d_A, d_out, n);

  GPU_CHECK(cudaMemcpy(out_ref, d_out_ref, out_size, cudaMemcpyDeviceToHost));
  GPU_CHECK(cudaMemcpy(out, d_out, out_size, cudaMemcpyDeviceToHost));
  bool error = false;
  for (size_t i = 0; i < n; i++) {
    unsigned char t = int(A[i]);
    if (out[i] != t) {
      printf("@%zu: out[%u] != %u\n", i, out[i], t);
      error = true;
      break;
    }
    if (out_ref[i] != t) {
      printf("@%zu: out_ref[%u] != %u\n", i, out_ref[i], t);
      error = true;
      break;
    }
  }
  printf("%s\n", error ? "FAIL" : "PASS");

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    reference<<<grid, block>>>(d_A, d_out_ref, n);
  }

  GPU_CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the reference kernel: %f (us)\n", (time * 1e-3f) / repeat);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    kernel<block_size, items_per_thread><<<grid, block>>>(d_A, d_out, n);
  }

  GPU_CHECK(cudaDeviceSynchronize());
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the blockAccess kernel: %f (us)\n", (time * 1e-3f) / repeat);

  GPU_CHECK(cudaFree(d_A));
  GPU_CHECK(cudaFree(d_out));
  GPU_CHECK(cudaFree(d_out_ref));
  free(A);
  free(out);
  free(out_ref);
  if (error) return 1;
  return 0;
}
