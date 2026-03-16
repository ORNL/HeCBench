#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cassert>
#include <cmath>
#include <sycl/sycl.hpp>

static void fill_values(float *r, sycl::half *a, sycl::half *b, size_t n) {
  for (size_t i = 0; i < n; ++i) {
    float ta = rand() % 2 ? -1.f : 1.f;
    float tb = rand() % 2 ? -1.f : 1.f;
    r[i] = ta + tb;
    a[i] = ta;
    b[i] = tb;
  }
}

static bool allclose(const float *ref, const sycl::half *got, size_t n,
                     float atol = 1e-2f) {
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

void base_elementwise_add_kernel(const sycl::half *__restrict__ gA,
                                 const sycl::half *__restrict__ gB,
                                 sycl::half *__restrict__ gC,
                                 const size_t n,
                                 sycl::nd_item<3> &item)
{
  size_t idx = item.get_global_id(2);
  if (idx < n) gC[idx] = gA[idx] + gB[idx];
}

void run_base(sycl::queue &q, const sycl::half *dA, const sycl::half *dB, sycl::half *dC,
              size_t total)
{
  const int TPB = 256;
  int grid = (total + TPB - 1) / TPB;
  sycl::range<3> gws (1, 1, grid * TPB);
  sycl::range<3> lws (1, 1, TPB);
  q.parallel_for(
    sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> item) {
      base_elementwise_add_kernel(dA, dB, dC, total, item);
  });
}


static const int vec_size = 8;

void vectorized_elementwise_add_kernel(const sycl::half *__restrict__ gA,
                                       const sycl::half *__restrict__ gB,
                                       sycl::half *__restrict__ gC,
                                       const size_t n,
                                 sycl::nd_item<3> &item)
{
  const int block_work_size = item.get_local_range(2) * vec_size;
  auto index = static_cast<size_t>(item.get_group(2)) * block_work_size +
               item.get_local_id(2) * vec_size;

  auto remaining = n - index;
  if (remaining < vec_size) {
    for (auto i = index; i < n; i++) {
      gC[i] = gA[i] + gB[i];
    }
  } else {

    auto a8 = *reinterpret_cast<const sycl::float4 *>(gA + index);
    auto b8 = *reinterpret_cast<const sycl::float4 *>(gB + index);

    auto a0 = *reinterpret_cast<sycl::half2 *>(&a8.x());
    auto a1 = *reinterpret_cast<sycl::half2 *>(&a8.y());
    auto a2 = *reinterpret_cast<sycl::half2 *>(&a8.z());
    auto a3 = *reinterpret_cast<sycl::half2 *>(&a8.w());
    auto b0 = *reinterpret_cast<sycl::half2 *>(&b8.x());
    auto b1 = *reinterpret_cast<sycl::half2 *>(&b8.y());
    auto b2 = *reinterpret_cast<sycl::half2 *>(&b8.z());
    auto b3 = *reinterpret_cast<sycl::half2 *>(&b8.w());

    sycl::float4 c8;
    *reinterpret_cast<sycl::half2 *>(&c8.x()) = a0 + b0;
    *reinterpret_cast<sycl::half2 *>(&c8.y()) = a1 + b1;
    *reinterpret_cast<sycl::half2 *>(&c8.z()) = a2 + b2;
    *reinterpret_cast<sycl::half2 *>(&c8.w()) = a3 + b3;
    *reinterpret_cast<sycl::float4 *>(gC + index) = c8;
  }
}

void run_vectorized(sycl::queue &q, const sycl::half *dA, const sycl::half *dB, sycl::half *dC,
                    size_t total)
{
  const int TPB = 256;
  int block_work_size = TPB * vec_size;
  int grid = (total + block_work_size - 1) / block_work_size;
  sycl::range<3> gws (1, 1, grid * TPB);
  sycl::range<3> lws (1, 1, TPB);
  q.parallel_for(
    sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> item) {
      vectorized_elementwise_add_kernel(dA, dB, dC, total, item);
  });
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

static const int TILE_M     = 16;   // block tile rows
static const int TILE_N     = 256;  // block tile cols  (= 32 lanes × 8 values)
static const int VALS_M     =  4;   // values per thread in M dimension
static const int VALS_N     =  8;   // values per thread in N dimension (128-bit load)
static const int WARPS      =  TILE_M / VALS_M;

template <int WARP_SIZE>
void tv_elementwise_add_kernel(const sycl::half *__restrict__ gA,
                               const sycl::half *__restrict__ gB,
                               sycl::half *__restrict__ gC,
                               const int M, const int N,
                               sycl::nd_item<3> &item)
{
  // Block-tile origin in global memory
  int blk_row = item.get_group(1) * TILE_M; // first row of this block's tile
  int blk_col = item.get_group(2) * TILE_N; // first col of this block's tile

  // Thread decomposition within the tile:
  int warp_id = item.get_local_id(2) / WARP_SIZE;
  int lane_id = item.get_local_id(2) % WARP_SIZE;

  // VALS_M rows for each warp
  int row_start = blk_row + warp_id * VALS_M;

  // A 128-bit (8×fp16) load for each thread
  int col_start = blk_col + lane_id * VALS_N;

  sycl::half a_frag[VALS_M][VALS_N];
  sycl::half b_frag[VALS_M][VALS_N];
  sycl::half c_frag[VALS_M][VALS_N];

  // merge three loops: load, compute, store
#pragma unroll
  for (int vm = 0; vm < VALS_M; ++vm) {
    int row = row_start + vm;
    // 128-bit (float4 = 8×fp16) vectorised load
    size_t base = (size_t)row * N + col_start;
    *reinterpret_cast<sycl::float4 *>(a_frag[vm]) =
        *reinterpret_cast<const sycl::float4 *>(gA + base);
    *reinterpret_cast<sycl::float4 *>(b_frag[vm]) =
        *reinterpret_cast<const sycl::float4 *>(gB + base);

    auto a = reinterpret_cast<const sycl::half2 *>(a_frag[vm]);
    auto b = reinterpret_cast<const sycl::half2 *>(b_frag[vm]);
    auto c = reinterpret_cast<sycl::half2 *>(c_frag[vm]);

#pragma unroll
    for (int i = 0; i < VALS_N / 2; i++)
      c[i] = a[i] + b[i];

    *reinterpret_cast<sycl::float4 *>(gC + base) =
        *reinterpret_cast<const sycl::float4 *>(c_frag[vm]);
  }
}

template <int WARP_SIZE>
void run_tv_layout(sycl::queue &q, const sycl::half *dA, const sycl::half *dB, sycl::half *dC,
                   int M, int N)
{
  assert(M % TILE_M == 0 && "M must be divisible by TILE_M=16");  // 4 warps
  assert(N % TILE_N == 0 && "N must be divisible by TILE_N=256"); // 1 warp

  int THREADS = WARPS * WARP_SIZE;
  sycl::range<3> gws (1, M / TILE_M, THREADS * N / TILE_N);
  sycl::range<3> lws (1, 1, THREADS);

  q.parallel_for(
    sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> item) {
      tv_elementwise_add_kernel<WARP_SIZE>(dA, dB, dC, M, N, item);
  });
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

int main(int argc, char *argv[]) try {
  if (argc != 4) {
    printf("Usage: %s <number of rows> <number of columns> <repeat>\n", argv[0]);
    return 1;
  }
  const int M = atoi(argv[1]);
  const int N = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  const size_t total   = (size_t)M * N;
  const size_t bytes = total * sizeof(sycl::half);

  sycl::half *hA = (sycl::half *)malloc(bytes);
  sycl::half *hB = (sycl::half *)malloc(bytes);
  sycl::half *hOut = (sycl::half *)malloc(bytes);
  float* hRef = (float*)malloc(total * sizeof(float));

  srand(42);
  fill_values(hRef, hA, hB, total);

  sycl::half *dA, *dB, *dC;
  dA = (sycl::half *)sycl::malloc_device(bytes, q);
  dB = (sycl::half *)sycl::malloc_device(bytes, q);
  dC = (sycl::half *)sycl::malloc_device(bytes, q);

  q.memcpy(dA, hA, bytes);
  q.memcpy(dB, hB, bytes);

  double bytes_moved = 3.0 * total * sizeof(sycl::half);

  printf("=== Base (one thread per element) ===\n");
  q.memset(dC, 0, bytes);
  run_base(q, dA, dB, dC, total);
  q.memcpy(hOut, dC, bytes).wait();
  bool pass1 = allclose(hRef, hOut, total);

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    run_base(q, dA, dB, dC, total);
  }
  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  float ms = time * 1e-6 / repeat;
  float gbps = (float)(bytes_moved / (ms * 1e-3) / 1e9);
  BenchResult r1 = {ms, gbps};
  print_result("base", pass1, r1, repeat);

  printf("=== Vectorised (8×fp16 = 128-bit load per thread) ===\n");
  q.memset(dC, 0, bytes);
  run_vectorized(q, dA, dB, dC, total);
  q.memcpy(hOut, dC, bytes).wait();
  bool pass2 = allclose(hRef, hOut, total);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    run_vectorized(q, dA, dB, dC, total);
  }
  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  ms = time * 1e-6 / repeat;
  gbps = (float)(bytes_moved / (ms * 1e-3) / 1e9);
  BenchResult r2 = {ms, gbps};
  print_result("vectorized", pass2, r2, repeat);

  printf("=== TV-layout (block tile 16×256, 8×fp16 per thread per row) ===\n");
  auto sg_sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
  auto r = std::max_element(sg_sizes.begin(), sg_sizes.end());
  int max_sg_size = *r;

  q.memset(dC, 0, bytes);
  if (max_sg_size == 64)
    run_tv_layout<64>(q, dA, dB, dC, M, N);
  else
    run_tv_layout<32>(q, dA, dB, dC, M, N);
  q.memcpy(hOut, dC, bytes).wait();
  bool pass3 = allclose(hRef, hOut, total);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    if (max_sg_size == 64)
      run_tv_layout<64>(q, dA, dB, dC, M, N);
    else
      run_tv_layout<32>(q, dA, dB, dC, M, N);
  }
  q.wait();
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

  sycl::free(dA, q);
  sycl::free(dB, q);
  sycl::free(dC, q);
  free(hA); free(hB); free(hRef); free(hOut);
  return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}
