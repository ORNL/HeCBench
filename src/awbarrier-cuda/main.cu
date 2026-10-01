// A software-pipelined (double-buffered) kernel that overlaps asynchronous 
// global->shared memory copies of the NEXT tile with the compute on the CURRENT tile.
//
// The kernel uses two cuda::barriers + cuda::memcpy_async to
// prefetch the next tile while computing the current one.
// The arrive()/wait() split is what lets the copy run in
// the background -> the load latency is hidden.


#include <chrono>
#include <cstdio>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cuda/barrier>

#define CHECK_CUDA(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,            \
              cudaGetErrorString(err__));                                      \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

namespace cg = cooperative_groups;

#ifndef TILE
#define TILE 256 // elements per tile == threads per block
#endif

// Compute-heavy transform applied to each element after it is staged in shared
// memory. Enough work that hiding the global-load latency actually matters.
__device__ __forceinline__ float heavy(float v, int numOps)
{
    #pragma unroll
    for (int k = 0; k < numOps; k++) {
        v = fmaf(v, 0.9999997f, 0.0000013f);
        v = fmaf(v, 1.0000003f, -0.0000007f);
    }
    return v;
}

// Synchronous version: load tile -> __syncthreads() -> compute.
// The load latency is exposed at the barrier on every tile.
__global__ void tileSync(const float *in, float *out, int numTiles, int numOps)
{
    __shared__ float s[TILE];

    for (int tile = blockIdx.x; tile < numTiles; tile += gridDim.x) {
        int gidx = tile * TILE + threadIdx.x;

        s[threadIdx.x] = in[gidx];
        __syncthreads();

        out[gidx] = heavy(s[threadIdx.x], numOps);
        __syncthreads();
    }
}

// Pipelined version: prefetch next tile with memcpy_async while computing the
// current tile. Two barriers implement the double buffer.
__global__ void tilePipelined(const float *in, float *out, int numTiles, int numOps)
{
#if __CUDA_ARCH__ >= 700
    auto block = cg::this_thread_block();

    __shared__ float s[2][TILE];
    __shared__ cuda::barrier<cuda::thread_scope_block> bar[2];

    if (block.thread_rank() == 0) {
        init(&bar[0], block.size());
        init(&bar[1], block.size());
    }
    block.sync();

    // Async copy for this block's first tile.
    int firstTile = blockIdx.x;
    if (firstTile < numTiles) {
        cuda::memcpy_async(block, &s[0][0], &in[firstTile * TILE], sizeof(float) * TILE, bar[0]);
    }

    int buf = 0;
    for (int tile = blockIdx.x; tile < numTiles; tile += gridDim.x) {
        int nextTile = tile + gridDim.x;
        int nbuf     = buf ^ 1;

        // Start loading the next tile into the other buffer
        if (nextTile < numTiles) {
            cuda::memcpy_async(block, &s[nbuf][0], &in[nextTile * TILE], sizeof(float) * TILE, bar[nbuf]);
        }

        // Wait only for the current tile's copy to complete, then compute on it.
        // The next tile's load is running during this compute.
        bar[buf].arrive_and_wait();

        out[tile * TILE + threadIdx.x] = heavy(s[buf][threadIdx.x], numOps);

        buf = nbuf;
    }
#endif
}

int main(int argc, char **argv)
{
    if (argc != 4) {
      printf("Usage: %s <number of tiles> <number of operations per work-item> <repeat>\n", argv[0]);
      return 1;
    }
    const int numTiles = atoi(argv[1]); //200000;
    const int numOps   = atoi(argv[2]); //200;
    const int iters    = atoi(argv[3]); //50;
    const int size     = numTiles * TILE;

    int major = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, 0));
    if (major < 7) {
        printf("This sample requires SM 7.0 or higher. Exiting...\n");
        return 1;
    }

    int smCount = 0;
    CHECK_CUDA(cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0));
    int numBlocks = smCount * 2;

    printf("Number of elements = %d, Number of tiles = %d, Number of operations per work-item = %d, Number of blocks = %d, iters = %d\n\n",
           size, TILE, numOps, numBlocks, iters);

    float *h_in, *d_in, *d_out, *h_outSync, *h_outPipe;
    CHECK_CUDA(cudaMallocHost(&h_in, sizeof(float) * size));
    CHECK_CUDA(cudaMallocHost(&h_outSync, sizeof(float) * size));
    CHECK_CUDA(cudaMallocHost(&h_outPipe, sizeof(float) * size));
    CHECK_CUDA(cudaMalloc(&d_in, sizeof(float) * size));
    CHECK_CUDA(cudaMalloc(&d_out, sizeof(float) * size));

    for (int i = 0; i < size; i++) {
        h_in[i] = (float)((i % 97) * 0.01f + 1.0f);
    }
    CHECK_CUDA(cudaMemcpy(d_in, h_in, sizeof(float) * size, cudaMemcpyHostToDevice));

    for (int it = 0; it < iters; it++) {
        tileSync<<<numBlocks, TILE>>>(d_in, d_out, numTiles, numOps); // warmup
    }
    CHECK_CUDA(cudaMemcpy(h_outSync, d_out, sizeof(float) * size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaDeviceSynchronize());
    auto start = std::chrono::steady_clock::now();

    for (int it = 0; it < iters; it++) {
        tileSync<<<numBlocks, TILE>>>(d_in, d_out, numTiles, numOps);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    double syncTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    syncTime = syncTime / iters;

    for (int it = 0; it < iters; it++) {
        tilePipelined<<<numBlocks, TILE>>>(d_in, d_out, numTiles, numOps); // warmup
    }
    CHECK_CUDA(cudaMemcpy(h_outPipe, d_out, sizeof(float) * size, cudaMemcpyDeviceToHost));

    CHECK_CUDA(cudaDeviceSynchronize());
    start = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; it++) {
        tilePipelined<<<numBlocks, TILE>>>(d_in, d_out, numTiles, numOps);
    }
    CHECK_CUDA(cudaDeviceSynchronize());
    end = std::chrono::steady_clock::now();
    double pipeTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    pipeTime = pipeTime / iters;

    int mismatches = 0;
    for (int i = 0; i < size; i++) {
        if (fabsf(h_outSync[i] - h_outPipe[i]) > 1e-3f) {
            if (mismatches < 5)
                printf("  mismatch at %d: sync=%f pipe=%f\n", i, h_outSync[i], h_outPipe[i]);
            mismatches++;
        }
    }
    printf("Correctness (sync vs pipelined): %s\n\n", mismatches == 0 ? "MATCH" : "MISMATCH");

    printf("Average kernel time over %d iterations:\n", iters);
    printf("  Synchronous (__syncthreads)      : %.4lf ms\n", syncTime * 1e-6);
    printf("  Pipelined (cuda::barrier + async): %.4lf ms\n", pipeTime * 1e-6);
    printf("  Speedup (sync / pipelined)       : %.3lfx\n", syncTime / pipeTime);

    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    CHECK_CUDA(cudaFreeHost(h_in));
    CHECK_CUDA(cudaFreeHost(h_outSync));
    CHECK_CUDA(cudaFreeHost(h_outPipe));

    bool ok = (mismatches == 0);
    printf("\n%s\n", ok ? "PASS" : "FAIL");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
