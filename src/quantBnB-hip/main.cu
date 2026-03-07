#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <hip/hip_runtime.h>
#include "block_load.h"
#include "block_store.h"
#include "code.h"

template <int STOCHASTIC>
__host__ __device__ uint8_t
dQuantize(float* smem_code, const float rand, float x)
{
    int pivot = 127;
    int upper_pivot = 255;
    int lower_pivot = 0;

    float lower = -1.0f;
    float upper = 1.0f;

    float val = smem_code[pivot];
    for(int i = 64; i > 0; i>>=1)
    {
        if(x > val)
        {
            lower_pivot = pivot;
            lower = val;
            pivot+=i;
        }
        else
        {
            upper_pivot = pivot;
            upper = val;
            pivot-=i;
        }
        val = smem_code[pivot];
    }

    if(upper_pivot == 255)
        upper = smem_code[upper_pivot];
    if(lower_pivot == 0)
        lower = smem_code[lower_pivot];

    if(!STOCHASTIC)
    {
      if(x > val)
      {
        float midpoint = (upper+val)*0.5f;
        if(x > midpoint)
        {
          return upper_pivot;
        }
        else
          return pivot;
      }
      else
      {
        float midpoint = (lower+val)*0.5f;
        if(x < midpoint)
          return lower_pivot;
        else
          return pivot;
      }
    }
    else
    {
      if(x > val)
      {
        float dist_to_upper = fabsf(upper-x);
        float dist_full = upper-val;
        if(rand >= dist_to_upper/dist_full) return upper_pivot;
        else return pivot;
      }
      else
      {
        float dist_to_lower = fabsf(lower-x);
        float dist_full = val-lower;
        if(rand >= dist_to_lower/dist_full) return lower_pivot;
        else return pivot;
      }
    }
}


#define NUM 4

template<int TH, int BLOCK_SIZE>
__global__ void kQuantize(const float *__restrict__ code,
                          const float * __restrict__ A,
                          uint8_t *out, const int n)
{
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;
  const int n_full = gridDim.x * BLOCK_SIZE;
  const int base_idx = bid * BLOCK_SIZE;

  float vals[NUM];
  uint8_t qvals[NUM];

  // 1D block of TH threads owning NUM items each
  typedef BlockLoad<float, TH, NUM> LoadFloat;
  typedef BlockStore<uint8_t, TH, NUM> StoreChar;

  __shared__ typename LoadFloat::TempStorage loadf_storage;
  __shared__ typename StoreChar::TempStorage storec_storage;
  __shared__ float smem_code[256];

  for (int i = tid; i < 256; i += blockDim.x)
  {
    smem_code[i] = code[i];
  }

  for (int i = base_idx; i < n; i += n_full)
  {
      int valid_items = min(n - i, BLOCK_SIZE);

      LoadFloat(loadf_storage).Load(&(A[i]), vals, valid_items);

      __syncthreads();

      #pragma unroll
      for(int j = 0; j < NUM; j++)
          qvals[j] = dQuantize<0>(smem_code, 0.0f, vals[j]);

      StoreChar(storec_storage).Store(&(out[i]), qvals, valid_items);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const size_t n = atol(argv[1]);
  const int repeat = atoi(argv[2]);

  const size_t A_size = n * sizeof(float);
  const size_t out_size = n * sizeof(uint8_t);
  const size_t code_size = 256 * sizeof(float); // code.h

  std::vector<float> A(n);
  std::vector<uint8_t> out(n), ref(n);

  std::mt19937 gen{19937};
 
  std::normal_distribution<float> d{0.f, 1.f};

  for (size_t i = 0; i < n; i++) {
    A[i] = d(gen); 
    ref[i] = dQuantize<0>(code, 0.f, A[i]);
  }

  float *d_A, *d_code;
  hipMalloc((void**)&d_A, A_size);
  hipMemcpy(d_A, A.data(), A_size, hipMemcpyHostToDevice);

  hipMalloc((void**)&d_code, code_size);
  hipMemcpy(d_code, code, code_size, hipMemcpyHostToDevice);

  uint8_t *d_out;
  hipMalloc((void**)&d_out, out_size);
  
  const int block_size = 256;

  dim3 grid ((n+block_size-1)/block_size);
  dim3 block (block_size);

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    kQuantize<block_size, block_size><<<grid, block>>>(d_code, d_A, d_out, n);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of kQuantize kernel with block size %d: %f (us)\n",
          block_size, (time * 1e-3f) / repeat);
  
  hipMemcpy(out.data(), d_out, out_size, hipMemcpyDeviceToHost);

  printf("%s\n", out == ref ? "PASS" : "FAIL");
  
  hipFree(d_A);
  hipFree(d_code);
  hipFree(d_out);
  if (out != ref) return 1;
  return 0;
}
