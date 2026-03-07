/* Reference: https://x.momo86.net/en?p=113 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#define NUM_OF_BLOCKS 1048576
#define NUM_OF_THREADS 256

__device__ half2 half_max(const half2 a, const half2 b) {
  const half2 sub = __hsub2(a, b);
  const unsigned sign = (*reinterpret_cast<const unsigned*>(&sub)) & 0x80008000u;
  const unsigned sw = 0x00003210 | (((sign >> 21) | (sign >> 13)) * 0x11);
  const unsigned int res = __byte_perm(*reinterpret_cast<const unsigned*>(&a), 
                                       *reinterpret_cast<const unsigned*>(&b), sw);
  return *reinterpret_cast<const half2*>(&res);
}

__device__ half half_max(const half a, const half b) {
  const half sub = __hsub(a, b);
  const unsigned sign = (*reinterpret_cast<const short*>(&sub)) & 0x8000u;
  const unsigned sw = 0x00000010 | ((sign >> 13) * 0x11);
  const unsigned short res = __byte_perm(*reinterpret_cast<const short*>(&a), 
                                         *reinterpret_cast<const short*>(&b), sw);
  return *reinterpret_cast<const half*>(&res);
}

template <typename T>
__global__
void hmax(T const *__restrict__ const a,
          T const *__restrict__ const b,
          T *__restrict__ const r,
          const size_t size)
{
  for (size_t i = threadIdx.x + blockDim.x * blockIdx.x; 
              i < size; i += blockDim.x * gridDim.x)
    r[i] = half_max(a[i], b[i]);
}


void generateInput(half2 * a, size_t size)
{
  for (size_t i = 0; i < size; ++i)
  {
    half2 temp;
    temp.x = static_cast<float>(rand() % 922021);
    temp.y = static_cast<float>(rand() % 922021);
    a[i] = temp;
  }
}

// compute the maximum of two values
int main(int argc, char *argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  size_t size = (size_t)NUM_OF_BLOCKS * NUM_OF_THREADS;

  const size_t size_bytes = size * sizeof(half2);

  half2 * a, *b, *r;
  half2 * d_a, *d_b, *d_r;

  a = (half2*) malloc (size_bytes);
  b = (half2*) malloc (size_bytes);
  r = (half2*) malloc (size_bytes);

  hipMalloc((void**)&d_a, size_bytes);
  hipMalloc((void**)&d_b, size_bytes);
  hipMalloc((void**)&d_r, size_bytes);

  // initialize input values
  srand(123); 
  generateInput(a, size);
  hipMemcpy(d_a, a, size_bytes, hipMemcpyHostToDevice);

  generateInput(b, size);
  hipMemcpy(d_b, b, size_bytes, hipMemcpyHostToDevice);

  for (int i = 0; i < repeat; i++)
    hmax<half2><<<NUM_OF_BLOCKS, NUM_OF_THREADS>>>(
      d_a, d_b, d_r, size);
  hipDeviceSynchronize();

  auto start = std::chrono::steady_clock::now();
  
  // run hmax2
  for (int i = 0; i < repeat; i++)
    hmax<half2><<<NUM_OF_BLOCKS, NUM_OF_THREADS>>>(
      d_a, d_b, d_r, size);

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (us)\n", (time * 1e-3f) / repeat);

  // verify
  hipMemcpy(r, d_r, size_bytes, hipMemcpyDeviceToHost);

  bool ok = true;
  for (size_t i = 0; i < size; ++i)
  {
    float2 fa = __half22float2(a[i]);
    float2 fb = __half22float2(b[i]);
    float2 fr = __half22float2(r[i]);
    float x = fmaxf(fa.x, fb.x);
    float y = fmaxf(fa.y, fb.y);
    if (fabsf(fr.x - x) > 1e-2 || fabsf(fr.y - y) > 1e-2) {
      ok = false;
      break;
    }
  }
  printf("fp16_hmax2 %s\n", ok ?  "PASS" : "FAIL");
  if (!ok) exit(1);

  for (int i = 0; i < repeat; i++)
    hmax<half><<<NUM_OF_BLOCKS, NUM_OF_THREADS>>>(
      (half*)d_a, (half*)d_b, (half*)d_r, size*2);
  hipDeviceSynchronize();

  start = std::chrono::steady_clock::now();
  
  // run hmax (the size is doubled)
  for (int i = 0; i < repeat; i++)
    hmax<half><<<NUM_OF_BLOCKS, NUM_OF_THREADS>>>(
      (half*)d_a, (half*)d_b, (half*)d_r, size*2);

  hipDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (us)\n", (time * 1e-3f) / repeat);

  hipMemcpy(r, d_r, size_bytes, hipMemcpyDeviceToHost);

  // verify
  ok = true;
  for (size_t i = 0; i < size; ++i)
  {
    float2 fa = __half22float2(a[i]);
    float2 fb = __half22float2(b[i]);
    float2 fr = __half22float2(r[i]);
    float x = fmaxf(fa.x, fb.x);
    float y = fmaxf(fa.y, fb.y);
    if (fabsf(fr.x - x) > 1e-3 || fabsf(fr.y - y) > 1e-3) {
      ok = false;
      break;
    }
  }

  printf("fp16_hmax %s\n", ok ?  "PASS" : "FAIL");
  if (!ok) exit(1);

  hipFree(d_a);
  hipFree(d_b);
  hipFree(d_r);
  free(a);
  free(b);
  free(r);

  return EXIT_SUCCESS;
}
