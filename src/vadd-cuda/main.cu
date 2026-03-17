#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cassert>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define GPU_CHECK(expr)                                               \
  do {                                                                \
    cudaError_t _err = (expr);                                        \
    if (_err != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error at %s:%d  %s\n",                    \
          __FILE__, __LINE__, cudaGetErrorString(_err));              \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  } while (0)

static void fill_values(float *r, __half* a, __half* b, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    float ta = rand() % 2 ? -1.f : 1.f;
    float tb = rand() % 2 ? -1.f : 1.f;
    r[i] = ta + tb;
    a[i] = ta;
    b[i] = tb;
  }
}

static bool allclose(const float* ref, const __half* got, size_t n, float atol = 1e-2f) {
  for (size_t i = 0; i < n; ++i) {
    float r = ref[i];
    float g = got[i];
    if (fabsf(r - g) > atol + 1e-5f * fabsf(r)) {
      fprintf(stderr, "  mismatch at [%zu]: ref=%.4f  got=%.4f\n", i, r, g);
      return false;
    }
  }
  return true;
}

__global__ void base_elementwise_add_kernel(
    const __half* __restrict__ gA,
    const __half* __restrict__ gB,
          __half* __restrict__ gC,
    const size_t n)
{
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) gC[idx] = gA[idx] + gB[idx];
}

void run_base(const __half* dA, const __half* dB, __half* dC, size_t total)
{
  constexpr int TPB = 256;
  int grid = (total + TPB - 1) / TPB;
  base_elementwise_add_kernel<<<grid, TPB>>>(dA, dB, dC, total);
}


static constexpr int vec_size = 8;

__global__ void vectorized_elementwise_add_kernel(
    const __half* __restrict__ gA,
    const __half* __restrict__ gB,
          __half* __restrict__ gC,
    const size_t n)
{
  const int block_work_size = blockDim.x * vec_size;
  auto index = static_cast<size_t>(blockIdx.x) * block_work_size + threadIdx.x * vec_size;

  auto remaining = n - index;
  if (remaining < vec_size) {
    for (auto i = index; i < n; i++) {
      gC[i] = gA[i] + gB[i];
    }
  } else {

    auto a8 = *reinterpret_cast<const float4*>(gA + index);
    auto b8 = *reinterpret_cast<const float4*>(gB + index);

    auto a0 = *reinterpret_cast<__half2*>(&a8.x);
    auto a1 = *reinterpret_cast<__half2*>(&a8.y);
    auto a2 = *reinterpret_cast<__half2*>(&a8.z);
    auto a3 = *reinterpret_cast<__half2*>(&a8.w);
    auto b0 = *reinterpret_cast<__half2*>(&b8.x);
    auto b1 = *reinterpret_cast<__half2*>(&b8.y);
    auto b2 = *reinterpret_cast<__half2*>(&b8.z);
    auto b3 = *reinterpret_cast<__half2*>(&b8.w);

    float4 c8;
    *reinterpret_cast<__half2*>(&c8.x) = __hadd2(a0, b0);
    *reinterpret_cast<__half2*>(&c8.y) = __hadd2(a1, b1);
    *reinterpret_cast<__half2*>(&c8.z) = __hadd2(a2, b2);
    *reinterpret_cast<__half2*>(&c8.w) = __hadd2(a3, b3);
    *reinterpret_cast<float4*>(gC + index) = c8;
  }
}


void run_vectorized(const __half* dA, const __half* dB, __half* dC, size_t total)
{
  constexpr int TPB = 256;
  int block_work_size = TPB * vec_size;
  int grid = (total + block_work_size - 1) / block_work_size;
  vectorized_elementwise_add_kernel<<<grid, TPB>>>(dA, dB, dC, total);
}

// ===========================================================================
// TV-layout element-wise add
//
//   Block tile: TILE_M=16 rows × TILE_N=256 cols  per thread-block.
//   Thread organisation inside the tile:
//     - 4 warps  (TILE_M / VALS_M  = 16/4  = 4 warps of 32 threads each)
//     - Threads in a warp are consecutive in the N direction (stride 1 in N)
//       matching the row-major layout for coalesced access.
//   Each thread loads:
//     - VALS_M=4 rows
//     - VALS_N=8 consecutive columns per row  (8×fp16 = 128-bit load)
//   Total values per thread = 4 × 8 = 32.
//   Total threads per block = 4 warps × 32 = 128.
//
// Memory access pattern:
//   Thread (warp_id, lane_id) handles rows
//     [warp_id*VALS_M .. warp_id*VALS_M + VALS_M)
//   and columns
//     [lane_id*VALS_N .. lane_id*VALS_N + VALS_N)
// ===========================================================================

static constexpr int WARP_SIZE  = 32;
static constexpr int TILE_M     = 16;   // block tile rows
static constexpr int TILE_N     = 256;  // block tile cols  (= 32 lanes × 8 values)
static constexpr int VALS_M     =  4;   // values per thread in M dimension
static constexpr int VALS_N     =  8;   // values per thread in N dimension (128-bit load)
static constexpr int WARPS      =  TILE_M / VALS_M;

__global__ void tv_elementwise_add_kernel(
    const __half* __restrict__ gA,
    const __half* __restrict__ gB,
          __half* __restrict__ gC,
    const int M,
    const int N)
{
  // Block-tile origin in global memory
  int blk_row = blockIdx.y * TILE_M;   // first row of this block's tile
  int blk_col = blockIdx.x * TILE_N;   // first col of this block's tile

  // Thread decomposition within the tile:
  int warp_id = threadIdx.x / WARP_SIZE;
  int lane_id = threadIdx.x % WARP_SIZE;

  // VALS_M rows for each warp
  int row_start = blk_row + warp_id * VALS_M;

  // A 128-bit (8×fp16) load for each thread
  int col_start = blk_col + lane_id * VALS_N;

  __half a_frag[VALS_M][VALS_N];
  __half b_frag[VALS_M][VALS_N];
  __half c_frag[VALS_M][VALS_N];

  // merge three loops: load, compute, store
#pragma unroll
  for (int vm = 0; vm < VALS_M; ++vm) {
    int row = row_start + vm;
    // 128-bit (float4 = 8×fp16) vectorised load
    size_t base = (size_t)row * N + col_start;
    *reinterpret_cast<float4*>(a_frag[vm]) = *reinterpret_cast<const float4*>(gA + base);
    *reinterpret_cast<float4*>(b_frag[vm]) = *reinterpret_cast<const float4*>(gB + base);

    auto a = reinterpret_cast<const __half2*>(a_frag[vm]);
    auto b = reinterpret_cast<const __half2*>(b_frag[vm]);
    auto c = reinterpret_cast<__half2*>(c_frag[vm]);

    #pragma unroll
    for (int i = 0; i < VALS_N / 2; i++)
      c[i] = __hadd2(a[i], b[i]);

    *reinterpret_cast<float4*>(gC + base) = *reinterpret_cast<const float4*>(c_frag[vm]);
  }
}

void run_tv_layout(const __half* dA, const __half* dB, __half* dC, int M, int N)
{
  assert(M % TILE_M == 0 && "M must be divisible by TILE_M=16");  // 4 warps
  assert(N % TILE_N == 0 && "N must be divisible by TILE_N=256"); // 1 warp
  int THREADS = WARPS * WARP_SIZE;
  dim3 grid(N / TILE_N, M / TILE_M);
  dim3 block(THREADS); // 128 threads
  tv_elementwise_add_kernel<<<grid, block>>>(dA, dB, dC, M, N);
}

// helpers
struct BenchResult {
  float avg_ms;   // mean over all timed iterations (ms)
  float gbps;     // effective memory bandwidth (GB/s) based on best time
};

static void print_result(const char* label, bool pass, BenchResult r, int repeats)
{
  printf("  correctness : %s\n",               pass ? "PASS" : "FAIL");
  printf("  avg  time   : %.3f ms  (%d iters)\n", r.avg_ms, repeats);
  printf("  bandwidth   : %.1f GB/s\n\n",       r.gbps);
}


int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <number of rows> <number of columns> <repeat>\n", argv[0]);
    return 1;
  }
  const int M = atoi(argv[1]);
  const int N = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const size_t total   = (size_t)M * N;
  const size_t bytes = (size_t)total * sizeof(__half);

  __half* hA   = (__half*)malloc(bytes);
  __half* hB   = (__half*)malloc(bytes);
  __half* hOut = (__half*)malloc(bytes);
  float* hRef = (float*)malloc(total * sizeof(float));

  srand(42);
  fill_values(hRef, hA, hB, total);

  __half *dA, *dB, *dC;
  GPU_CHECK(cudaMalloc(&dA, bytes));
  GPU_CHECK(cudaMalloc(&dB, bytes));
  GPU_CHECK(cudaMalloc(&dC, bytes));
  GPU_CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
  GPU_CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

  double bytes_moved = 3.0 * total * sizeof(__half);

  printf("=== Base (one thread per element) ===\n");
  GPU_CHECK(cudaMemset(dC, 0, bytes));
  run_base(dA, dB, dC, total);
  GPU_CHECK(cudaMemcpy(hOut, dC, bytes, cudaMemcpyDeviceToHost));
  bool pass1 = allclose(hRef, hOut, total);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    run_base(dA, dB, dC, total);
  }
  GPU_CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  float ms = time * 1e-6 / repeat;
  float gbps = (float)(bytes_moved / (ms * 1e-3) / 1e9);
  BenchResult r1 = {ms, gbps};
  print_result("base", pass1, r1, repeat);

  printf("=== Vectorised (8×fp16 = 128-bit load per thread) ===\n");
  GPU_CHECK(cudaMemset(dC, 0, bytes));
  run_vectorized(dA, dB, dC, total);
  GPU_CHECK(cudaMemcpy(hOut, dC, bytes, cudaMemcpyDeviceToHost));
  bool pass2 = allclose(hRef, hOut, total);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    run_vectorized(dA, dB, dC, total);
  }
  GPU_CHECK(cudaDeviceSynchronize());
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  ms = time * 1e-6 / repeat;
  gbps = (float)(bytes_moved / (ms * 1e-3) / 1e9);
  BenchResult r2 = {ms, gbps};
  print_result("vectorized", pass2, r2, repeat);

  printf("=== TV-layout (block tile 16×256, 8×fp16 per thread per row) ===\n");
  GPU_CHECK(cudaMemset(dC, 0, bytes));
  run_tv_layout(dA, dB, dC, M, N);
  GPU_CHECK(cudaMemcpy(hOut, dC, bytes, cudaMemcpyDeviceToHost));
  bool pass3 = allclose(hRef, hOut, total);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    run_tv_layout(dA, dB, dC, M, N);
  }
  GPU_CHECK(cudaDeviceSynchronize());
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  ms = time * 1e-6 / repeat;
  gbps = (float)(bytes_moved / (ms * 1e-3) / 1e9);
  BenchResult r3 = {ms, gbps};
  print_result("tv_layout", pass3, r3, repeat);

  printf("─────────────────────────────────────────────────────\n");
  printf("%-12s  %9s  %10s\n", "Kernel", "Avg(ms)", "BW(GB/s)");
  printf("─────────────────────────────────────────────────────\n");
  printf("%-12s  %9.3f  %10.1f\n", "base",       r1.avg_ms, r1.gbps);
  printf("%-12s  %9.3f  %10.1f\n", "vectorized", r2.avg_ms, r2.gbps);
  printf("%-12s  %9.3f  %10.1f\n", "tv_layout",  r3.avg_ms, r3.gbps);
  printf("─────────────────────────────────────────────────────\n");

  GPU_CHECK(cudaFree(dA));
  GPU_CHECK(cudaFree(dB));
  GPU_CHECK(cudaFree(dC));
  free(hA); free(hB); free(hRef); free(hOut);
  return 0;
}
