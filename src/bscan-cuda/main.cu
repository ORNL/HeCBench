//-----------------------------------------------------------------------
// Reference
//
// Harris, M. and Garland, M., 2012.
// Optimizing parallel prefix operations for the Fermi architecture.
// In GPU Computing Gems Jade Edition (pp. 29-38). Morgan Kaufmann.
//-----------------------------------------------------------------------

#include <cstdio>
#include <cstring>
#include <chrono>
#include <cuda.h>
#include <cstdlib>

__device__ __inline__ int warp_scan(int val, volatile int *s_data)
{
  // initialize shared memory accessed by each warp with zeros
  int idx = 2 * threadIdx.x - (threadIdx.x & (warpSize-1));
  s_data[idx] = 0;
  idx += warpSize;
  int t = s_data[idx] = val;
  s_data[idx] = t += s_data[idx - 1];
  s_data[idx] = t += s_data[idx - 2];
  s_data[idx] = t += s_data[idx - 4];
  s_data[idx] = t += s_data[idx - 8];
  s_data[idx] = t += s_data[idx -16];
  return s_data[idx-1];
}

__device__ __inline__ unsigned int lanemask_lt()
{
#ifdef ASM
  unsigned int mask;
  asm("mov.u32 %0, %lanemask_lt;" : "=r"(mask));
  return mask;
#else
  const unsigned int lane = threadIdx.x & (warpSize-1);
  return (1 << (lane)) - 1;
#endif
}

// warp scan optimized for binary
__device__ __inline__ unsigned int binary_warp_scan(bool p)
{
  const unsigned int mask = lanemask_lt();
#if (CUDART_VERSION < 9000)
  unsigned int b = __ballot(p);
  return __popc(b & mask);
#else
  unsigned int b = __ballot_sync(mask, p);
  return __popc(b);
#endif
}

// positive numbers
__host__ __device__ __inline__
bool valid(int x) {
  return x > 0;
}

__device__ __inline__ int block_binary_prefix_sums(int x)
{
  // 2 x warpIdx's upper bound (1024/32)
  __shared__ int sdata[64];

  bool predicate = valid(x);

  // A. Compute exclusive prefix sums within each warp
  int warpPrefix = binary_warp_scan(predicate);
  int idx = threadIdx.x;
  int warpIdx = idx / warpSize;
  int laneIdx = idx & (warpSize - 1);
#ifdef DEBUG
  printf("A %d %d %d\n", warpIdx, laneIdx, warpPrefix);
#endif

  // B. The last thread of each warp stores inclusive
  // prefix sum to the warp’s index in shared memory
  if (laneIdx == warpSize - 1) {
    sdata[warpIdx] = warpPrefix + predicate;
#ifdef DEBUG
    printf("B %d %d\n", warpIdx, sdata[warpIdx]);
#endif
  }
  __syncthreads();

  // C. One warp scans the warp partial sums
  if (idx < warpSize) {
    sdata[idx] = warp_scan(sdata[idx], sdata);
#ifdef DEBUG
    printf("C: %d %d\n", idx, sdata[idx]);
#endif
  }
  __syncthreads();

  // D. Each thread adds prefix sums of warp partial
  // sums to its own intra−warp prefix sums
  return warpPrefix + sdata[warpIdx];
}

__global__ void binary_scan(
        int *__restrict__ g_odata,
  const int *__restrict__ g_idata)
{
  int i = threadIdx.x;
  g_odata[i] = block_binary_prefix_sums(g_idata[i]);
}

template <int N>
void bscan (const int repeat) 
{
  int h_in[N];
  int h_out[N];
  int ref_out[N];

  int *d_in, *d_out;
  cudaMalloc((void**)&d_in, N*sizeof(int));
  cudaMalloc((void**)&d_out, N*sizeof(int));

  bool ok = true;
  double time = 0.0;
  srand(123);

  size_t grid_size = 12*7*8*9*10;
  dim3 grids (grid_size);
  dim3 blocks (N);

  int valid_count = 0;

  for (int i = 0; i < repeat; i++) {
    for (int n = 0; n < N; n++) {
      h_in[n] = rand() % N - N/2;
      if (valid(h_in[n])) valid_count++;  // total number of valid elements
    }
    cudaMemcpy(d_in, h_in, N*sizeof(int), cudaMemcpyHostToDevice); 

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    binary_scan<<<grids, blocks>>>(d_out, d_in);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // verify exclusive sum
    cudaMemcpy(h_out, d_out, N*sizeof(int), cudaMemcpyDeviceToHost);

    ref_out[0] = 0;
    ok &= (h_out[0] == ref_out[0]);
    for (int i = 1; i < N; i++) {
      ref_out[i] = ref_out[i-1] + (h_in[i-1] > 0);
      ok &= (ref_out[i] == h_out[i]);
    }
    if (!ok) break;
  } // for

  printf("Block size = %d, ratio of valid elements = %f, verify = %s\n",
          N, valid_count * 1.f / (N * repeat), ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  if (ok) {
    printf("Average execution time: %f (us)\n", (time * 1e-3f) / repeat);
    printf("Billion elements per second: %f\n\n",
            grid_size * N * repeat / time);
  }

  cudaFree(d_in);
  cudaFree(d_out);
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);
    
  // scan over N elements (N = [32, 1024])
  bscan<32>(repeat);
  bscan<64>(repeat);
  bscan<128>(repeat);
  bscan<256>(repeat);
  bscan<512>(repeat);
  bscan<1024>(repeat);

  return 0; 
}
