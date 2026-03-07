#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <hip/hip_runtime.h>

template<typename T>
void verify(const T* cpu_out, const T* gpu_out, int64_t n)
{
  int error = memcmp(cpu_out, gpu_out, n * sizeof(T));
  if (error) {
    for (int64_t i = 0; i < n; i++) {
      if (cpu_out[i] != gpu_out[i]) {
        printf("@%zu: %lf != %lf\n", i, (double)cpu_out[i], (double)gpu_out[i]);
        break;
      }
    }
  }
  printf("%s\n", error ? "FAIL" : "PASS");
}

// bank conflict aware optimization

#define LOG_MEM_BANKS 5
#define OFFSET(n) ((n) >> LOG_MEM_BANKS)

// N is the number of elements to scan in a thread block
template<typename T, int N>
__global__ void scan_bcao (
  const int64_t nblocks,
        T *__restrict__ g_odata,
  const T *__restrict__ g_idata)
{
  __shared__ T temp[2*N];

  for (int64_t bid = blockIdx.x; bid < nblocks; bid += gridDim.x)
  {
    auto gi = g_idata + bid * N;
    auto go = g_odata + bid * N;

    int thid = threadIdx.x;
    int a = thid;
    int b = a + (N/2);
    int oa = OFFSET(a);
    int ob = OFFSET(b);

    temp[a + oa] = gi[a];
    temp[b + ob] = gi[b];

    int offset = 1;
    for (int d = N >> 1; d > 0; d >>= 1)
    {
      __syncthreads();
      if (thid < d)
      {
        int ai = offset*(2*thid+1)-1;
        int bi = offset*(2*thid+2)-1;
        ai += OFFSET(ai);
        bi += OFFSET(bi);
        temp[bi] += temp[ai];
      }
      offset *= 2;
    }

    if (thid == 0) temp[N-1+OFFSET(N-1)] = 0; // clear the last elem
    for (int d = 1; d < N; d *= 2) // traverse down
    {
      offset >>= 1;
      __syncthreads();
      if (thid < d)
      {
        int ai = offset*(2*thid+1)-1;
        int bi = offset*(2*thid+2)-1;
        ai += OFFSET(ai);
        bi += OFFSET(bi);
        T t = temp[ai];
        temp[ai] = temp[bi];
        temp[bi] += t;
      }
    }
    __syncthreads(); // required

    go[a] = temp[a + oa];
    go[b] = temp[b + ob];
  }
}

template<typename T, int N>
__global__ void scan(
  const int64_t nblocks,
        T *__restrict__ g_odata,
  const T *__restrict__ g_idata)
{
  __shared__ T temp[N];

  for (int64_t bid = blockIdx.x; bid < nblocks; bid += gridDim.x)
  {
    auto gi = g_idata + bid * N;
    auto go = g_odata + bid * N;

    int thid = threadIdx.x;
    int offset = 1;
    temp[2*thid]   = gi[2*thid];
    temp[2*thid+1] = gi[2*thid+1];
    for (int d = N >> 1; d > 0; d >>= 1)
    {
      __syncthreads();
      if (thid < d)
      {
        int ai = offset*(2*thid+1)-1;
        int bi = offset*(2*thid+2)-1;
        temp[bi] += temp[ai];
      }
      offset *= 2;
    }

    if (thid == 0) temp[N-1] = 0; // clear the last elem
    for (int d = 1; d < N; d *= 2) // traverse down
    {
      offset >>= 1;
      __syncthreads();
      if (thid < d)
      {
        int ai = offset*(2*thid+1)-1;
        int bi = offset*(2*thid+2)-1;
        T t = temp[ai];
        temp[ai] = temp[bi];
        temp[bi] += t;
      }
    }
    go[2*thid] = temp[2*thid];
    go[2*thid+1] = temp[2*thid+1];
  }
}

template <typename T, int N>
void runTest (const int64_t n, const int repeat, bool timing = false)
{
  int64_t num_blocks = (n + N - 1) / N;

  int64_t nelems = num_blocks * N; // actual total number of elements

  int64_t bytes = nelems * sizeof(T);

  T *in = (T*) malloc (bytes);
  T *cpu_out = (T*) malloc (bytes);
  T *gpu_out = (T*) malloc (bytes);

  srand(123);
  for (int64_t n = 0; n < nelems; n++) in[n] = rand() % 5 + 1;

  T *t_in = in;
  T *t_out = cpu_out;
  for (int64_t n = 0; n < num_blocks; n++) {
    t_out[0] = 0;
    for (int i = 1; i < N; i++)
      t_out[i] = t_out[i-1] + t_in[i-1];
    t_out += N;
    t_in += N;
  }

  T *d_in, *d_out;

  hipMalloc((void**)&d_in, bytes);
  hipMemcpy(d_in, in, bytes, hipMemcpyHostToDevice);

  hipMalloc((void**)&d_out, bytes);

  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, 0);
  dim3 grids (16 * prop.multiProcessorCount);
  dim3 blocks (N/2);

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    scan<T, N><<<grids, blocks>>>(num_blocks, d_out, d_in);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  if (timing) {
    printf("Element size in bytes is %zu. Average execution time of scan (w/  bank conflicts): %f (us)\n",
           sizeof(T), (time * 1e-3f) / repeat);
  }
  hipMemcpy(gpu_out, d_out, bytes, hipMemcpyDeviceToHost);
  if (!timing) verify(cpu_out, gpu_out, nelems);

  // bcao
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    scan_bcao<T, N><<<grids, blocks>>>(num_blocks, d_out, d_in);
  }

  hipDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  auto bcao_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  if (timing) {
    printf("Element size in bytes is %zu. Average execution time of scan (w/o bank conflicts): %f (us). ",
           sizeof(T), (bcao_time * 1e-3f) / repeat);
    printf("Reduce the time by %.1f%%\n", (time - bcao_time) * 1.0 / time * 100);
  }
  hipMemcpy(gpu_out, d_out, bytes, hipMemcpyDeviceToHost);
  if (!timing) verify(cpu_out, gpu_out, nelems);

  hipFree(d_in);
  hipFree(d_out);
  free(in);
  free(cpu_out);
  free(gpu_out);
}

template<int N>
void run (const int64_t n, const int repeat) {
  for (int i = 0; i < 2; i++) {
    bool report_timing = i > 0;
    printf("\nThe number of elements to scan in a thread block: %d\n", N);
    runTest< char, N>(n, repeat, report_timing);
    runTest<short, N>(n, repeat, report_timing);
    runTest<  int, N>(n, repeat, report_timing);
    runTest< long, N>(n, repeat, report_timing);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int64_t n = atol(argv[1]);
  const int repeat = atoi(argv[2]);

  run< 128>(n, repeat);
  run< 256>(n, repeat);
  run< 512>(n, repeat);
  run<1024>(n, repeat);
  run<2048>(n, repeat);

  if (error) return 1;
  return 0;
}
