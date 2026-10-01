/*
 * Asynchronous (stream-ordered) memory allocation benchmark.
 *
 * Real use case modelled here: a streaming / batched inference-style service
 * that receives a queue of independent "jobs" of *varying* problem sizes and
 * processes them concurrently over a small pool of CUDA streams.  Each job
 * needs temporary device scratch buffers whose size depends on the job, so the
 * buffers have to be allocated and freed on every job.
 *
 * This pattern is common:
 *   - deep-learning inference serving with variable batch / sequence lengths,
 *   - graph analytics (per-iteration frontier buffers of changing size),
 *   - sparse linear algebra / batched solvers,
 *   - image / signal processing pipelines.
 *
 * Two implementations of the exact same work are compared:
 *
 *   1. Synchronous allocation  (cudaMalloc / cudaFree)
 *      Every cudaMalloc/cudaFree implicitly synchronizes the whole device,
 *      which serializes the streams and pays the OS allocation cost on every
 *      job.
 *
 *   2. Asynchronous allocation (cudaMallocAsync / cudaFreeAsync)
 *      Allocations are stream-ordered and served from a memory pool.  With the
 *      pool release threshold raised, freed memory stays in the pool and is
 *      recycled across jobs, so after warm-up there is no OS round-trip and no
 *      device-wide synchronization -> the streams overlap.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <climits>
#include <chrono>
#include <cmath>
#include <vector>
#include <random>
#include <iostream>
#include <cuda_runtime.h>

#define checkCudaErrors(err) __checkCudaErrors(err, __FILE__, __LINE__)
inline void __checkCudaErrors(cudaError_t err, const char *file, const int line) {
  if (cudaSuccess != err) {
    std::cerr << "CUDA Error = " << err << ": " << cudaGetErrorString(err)
              << " from file " << file << ", line " << line << std::endl;
    exit(EXIT_FAILURE);
  }
}

#define NSTREAMS 4

// A non-trivial, deterministic per-element transform so that the compute
// kernels take a meaningful amount of time (making stream overlap matter) and
// the results can be verified.
__global__ void process(const float *__restrict__ in, float *__restrict__ out,
                         int n, int iter) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = in[i];
    for (int k = 0; k < iter; k++) {
      v = v * 0.9999f + 0.0001f;
      v = sinf(v) * cosf(v) + sqrtf(fabsf(v) + 1.0f);
    }
    out[i] = v;
  }
}

// Host reference used to spot-check a few elements.
static float process_ref(float v, int iter) {
  for (int k = 0; k < iter; k++) {
    v = v * 0.9999f + 0.0001f;
    v = sinf(v) * cosf(v) + sqrtf(fabsf(v) + 1.0f);
  }
  return v;
}

// Synchronous allocation: cudaMalloc / cudaFree per job.
double run_sync(const float *h_in, float *h_out,
                const std::vector<size_t> &sizes,
                const std::vector<size_t> &offsets,
                cudaStream_t *streams, int iter) {
  auto start = std::chrono::steady_clock::now();
  for (size_t j = 0; j < sizes.size(); j++) {
    const int n = (int)sizes[j];
    const size_t bytes = (size_t)n * sizeof(float);
    cudaStream_t s = streams[j % NSTREAMS];

    float *d_in, *d_out;
    checkCudaErrors(cudaMalloc(&d_in, bytes));
    checkCudaErrors(cudaMalloc(&d_out, bytes));

    checkCudaErrors(cudaMemcpyAsync(d_in, h_in + offsets[j], bytes,
                                    cudaMemcpyHostToDevice, s));
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    process<<<grid, block, 0, s>>>(d_in, d_out, n, iter);
    checkCudaErrors(cudaMemcpyAsync(h_out + offsets[j], d_out, bytes,
                                    cudaMemcpyDeviceToHost, s));

    checkCudaErrors(cudaFree(d_in));
    checkCudaErrors(cudaFree(d_out));
  }
  checkCudaErrors(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

// Asynchronous (stream-ordered) allocation: cudaMallocAsync / cudaFreeAsync.
double run_async(const float *h_in, float *h_out,
                 const std::vector<size_t> &sizes,
                 const std::vector<size_t> &offsets,
                 cudaStream_t *streams, int iter) {
  auto start = std::chrono::steady_clock::now();
  for (size_t j = 0; j < sizes.size(); j++) {
    const int n = (int)sizes[j];
    const size_t bytes = (size_t)n * sizeof(float);
    cudaStream_t s = streams[j % NSTREAMS];

    float *d_in, *d_out;
    checkCudaErrors(cudaMallocAsync(&d_in, bytes, s));
    checkCudaErrors(cudaMallocAsync(&d_out, bytes, s));

    checkCudaErrors(cudaMemcpyAsync(d_in, h_in + offsets[j], bytes,
                                    cudaMemcpyHostToDevice, s));
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    process<<<grid, block, 0, s>>>(d_in, d_out, n, iter);
    checkCudaErrors(cudaMemcpyAsync(h_out + offsets[j], d_out, bytes,
                                    cudaMemcpyDeviceToHost, s));

    checkCudaErrors(cudaFreeAsync(d_in, s));
    checkCudaErrors(cudaFreeAsync(d_out, s));
  }
  checkCudaErrors(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

int main(int argc, char **argv) {
  int num_jobs = (argc > 1) ? atoi(argv[1]) : 2000;
  int iter     = (argc > 2) ? atoi(argv[2]) : 50;

  int dev = 0;
  int isMemPoolSupported = 0;
  checkCudaErrors(cudaDeviceGetAttribute(&isMemPoolSupported,
                                         cudaDevAttrMemoryPoolsSupported, dev));
  if (!isMemPoolSupported) {
    printf("Skip execution as device does not support Memory Pools\n");
    return 1;
  }

  // Jobs of varying sizes (the source of the repeated, differently-sized
  // allocations that stress the allocator).
  const size_t minN = 1 << 15;  //  32768 elements
  const size_t maxN = 1 << 17;  // 131072 elements
  std::mt19937 rng(123);
  std::uniform_int_distribution<size_t> dist(minN, maxN);

  std::vector<size_t> sizes(num_jobs);
  std::vector<size_t> offsets(num_jobs);
  size_t total = 0;
  for (int j = 0; j < num_jobs; j++) {
    sizes[j] = dist(rng);
    offsets[j] = total;
    total += sizes[j];
  }

  printf("Streaming pipeline: %d jobs over %d streams, sizes [%zu, %zu], "
         "%d compute iterations/element\n",
         num_jobs, NSTREAMS, minN, maxN, iter);
  printf("Total working set: %.1f MB per array\n",
         total * sizeof(float) / (1024.0 * 1024.0));

  float *h_in, *h_out_sync, *h_out_async;
  checkCudaErrors(cudaMallocHost(&h_in, total * sizeof(float)));
  checkCudaErrors(cudaMallocHost(&h_out_sync, total * sizeof(float)));
  checkCudaErrors(cudaMallocHost(&h_out_async, total * sizeof(float)));

  for (int j = 0; j < num_jobs; j++)
    for (size_t i = 0; i < sizes[j]; i++)
      h_in[offsets[j] + i] = sinf((float)j * 0.001f + (float)i * 1e-6f);

  cudaStream_t streams[NSTREAMS];
  for (int i = 0; i < NSTREAMS; i++)
    checkCudaErrors(cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking));

  // Raise the pool release threshold so freed memory is recycled instead of
  // being returned to the OS on synchronization. This is the key knob that
  // lets the pool reach a steady state.
  cudaMemPool_t memPool;
  checkCudaErrors(cudaDeviceGetDefaultMemPool(&memPool, dev));
  uint64_t thresholdVal = UINT64_MAX;
  checkCudaErrors(cudaMemPoolSetAttribute(
      memPool, cudaMemPoolAttrReleaseThreshold, (void *)&thresholdVal));

  // Warm up both paths (context init + pool steady state), untimed.
  run_async(h_in, h_out_async, sizes, offsets, streams, iter);
  run_sync(h_in, h_out_sync, sizes, offsets, streams, iter);

  double t_async = run_async(h_in, h_out_async, sizes, offsets, streams, iter);
  double t_sync  = run_sync(h_in, h_out_sync, sizes, offsets, streams, iter);

  printf("\nSynchronous  (cudaMalloc / cudaFree)          : %8.2f ms\n", t_sync);
  printf("Asynchronous (cudaMallocAsync / cudaFreeAsync): %8.2f ms\n", t_async);
  printf("Speedup from asynchronous allocation          : %8.2fx\n",
         t_sync / t_async);

  // Verify: async output must match sync output exactly, and both must match
  // a host reference on sampled elements.
  bool ok = (memcmp(h_out_sync, h_out_async, total * sizeof(float)) == 0);
  for (int j = 0; j < num_jobs && ok; j += (num_jobs / 16 + 1)) {
    for (size_t i = 0; i < sizes[j]; i += (sizes[j] / 8 + 1)) {
      float ref = process_ref(h_in[offsets[j] + i], iter);
      if (fabsf(ref - h_out_async[offsets[j] + i]) > 1e-3f) { ok = false; break; }
    }
  }
  printf("\n%s\n", ok ? "PASS" : "FAIL");

  for (int i = 0; i < NSTREAMS; i++)
    checkCudaErrors(cudaStreamDestroy(streams[i]));
  checkCudaErrors(cudaFreeHost(h_in));
  checkCudaErrors(cudaFreeHost(h_out_sync));
  checkCudaErrors(cudaFreeHost(h_out_async));

  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
