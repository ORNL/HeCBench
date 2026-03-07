#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <omp.h>
#include "code.h"

template <int STOCHASTIC>
uint8_t
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

  float* A = (float*) malloc (A_size);
  uint8_t* out = (uint8_t*) malloc (out_size);
  uint8_t* ref = (uint8_t*) malloc (out_size);

  std::mt19937 gen{19937};
 
  std::normal_distribution<float> d{0.f, 1.f};

  for (size_t i = 0; i < n; i++) {
    A[i] = d(gen);
    ref[i] = dQuantize<0>(code, 0.f, A[i]);
  }

  const int block_size = 256;
  const int grid = ((n+block_size-1)/block_size);

  #pragma omp target data map(to: A[0:n], code[0:256]) \
                          map(from: out[0:n])
  {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      #pragma omp target teams distribute parallel for num_teams(grid) num_threads(block_size)
      for (size_t i = 0; i < n; i++) {
        out[i] = dQuantize<0>(code, 0.f, A[i]);
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kQuantize kernel with block size %d: %f (us)\n",
            block_size, (time * 1e-3f) / repeat);
  }

  printf("%s\n", memcmp(out, ref, out_size) ? "FAIL" : "PASS");
  if (out_size)) exit(1);

  free(A);
  free(out);
  free(ref);
  return 0;
}
