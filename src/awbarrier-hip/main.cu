// A software-pipelined (double-buffered) kernel that overlaps asynchronous
// global->shared (LDS) memory copies of the NEXT tile with the compute on the
// CURRENT tile.
// async global-LDS copy buffer_load_dword uses the m0 register for the per-wave LDS destination
// base and s_waitcnt vmcnt as the completion fence.
// Double-buffering: at most two async copies are outstanding, so vmcnt(1) drains only the current tile's copy
// while the next tile's copy keeps running during compute; vmcnt(0) handles the final tile.
//  __builtin_amdgcn_s_barrier() replaces block.sync().
// The async path assumes a wavefront size of 64 (checked at runtime), and tileSync is a straight port.
//
// Reference https://github.com/carlushuang/gcnasmhttps://github.com/carlushuang/gcnasm


#include <chrono>
#include <cstdio>
#include <cmath>
#include <hip/hip_runtime.h>

#define CHECK_HIP(call)                                                        \
  do {                                                                         \
    hipError_t err__ = (call);                                                 \
    if (err__ != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error %s:%d: %s\n", __FILE__, __LINE__,             \
              hipGetErrorString(err__));                                       \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

#ifndef TILE
#define TILE 256 // elements per tile == threads per block
#endif

#define WAVE_SIZE 64

using int32x4_t = int32_t __attribute__((ext_vector_type(4)));
using index_t = int;

// Build a buffer resource descriptor (V#) for the async global->LDS loads.
#define BUFFER_RSRC_WORD3 0x00020000
struct buffer_resource {
    const void *ptr;
    uint32_t range;
    uint32_t config;
};
__device__ __forceinline__ int32x4_t make_buffer_resource(const void *ptr,
                                                           uint32_t range = 0xffffffff) {
    buffer_resource res{ptr, range, BUFFER_RSRC_WORD3};
    return __builtin_bit_cast(int32x4_t, res);
}

// Issue an asynchronous global->LDS copy of one dword for each active lane.
// The LDS destination base is taken from the m0 register (set separately);
// lane i writes to m0 + i * 4 bytes.
__device__ __forceinline__ void async_load_lds_dword(void *smem, int32x4_t rsrc,
                                                      index_t voffset) {
    asm volatile("buffer_load_dword %1, %2, 0 offen offset:0 lds"
                 : "=r"(smem) /* dummy dependency for smem */
                 : "v"(voffset), "s"(rsrc)
                 : "memory");
}

// Set the m0 register (LDS write base for the async copies).
__device__ __forceinline__ void m0_set(uint32_t v) {
    asm volatile("s_mov_b32 m0, %0" : : "s"(v) : "memory");
}

// Wait until at most 'cnt' vector-memory operations remain outstanding.
template <int cnt>
__device__ __forceinline__ void vmcnt_fence() {
    asm volatile("s_waitcnt vmcnt(%0)" : : "n"(cnt) : "memory");
}

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

// Pipelined version: prefetch next tile with an async global->LDS copy while
// computing the current tile. Two LDS buffers implement the double buffer, and
// s_waitcnt vmcnt() tracks the in-flight copies: at most two copies are ever
// outstanding, so waiting for vmcnt(1) drains only the CURRENT tile's copy
// while the NEXT tile's copy keeps running during the compute.
__global__ void tilePipelined(const float *in, float *out, int numTiles, int numOps)
{
    __shared__ float s[2][TILE];

    const int tid  = threadIdx.x;
    const int wave = tid / WAVE_SIZE;

    // LDS byte offset of the per-wave write region for each buffer.
    const uint32_t waveOff = wave * WAVE_SIZE * (uint32_t)sizeof(float);
    const uint32_t ldsBase[2] = {
        (uint32_t)(uintptr_t)(&s[0][0]) + waveOff,
        (uint32_t)(uintptr_t)(&s[1][0]) + waveOff,
    };

    // Async copy for this block's first tile into buffer 0.
    int firstTile = blockIdx.x;
    if (firstTile < numTiles) {
        m0_set(__builtin_amdgcn_readfirstlane(ldsBase[0]));
        async_load_lds_dword(&s[0][0], make_buffer_resource(&in[firstTile * TILE]),
                             tid * (int)sizeof(float));
    }

    int buf = 0;
    for (int tile = blockIdx.x; tile < numTiles; tile += gridDim.x) {
        int nextTile = tile + gridDim.x;
        int nbuf     = buf ^ 1;

        // Start loading the next tile into the other buffer.
        if (nextTile < numTiles) {
            m0_set(__builtin_amdgcn_readfirstlane(ldsBase[nbuf]));
            async_load_lds_dword(&s[nbuf][0], make_buffer_resource(&in[nextTile * TILE]),
                                 tid * (int)sizeof(float));
            // Two copies outstanding: wait until only the next one remains,
            // i.e. the current tile's copy has completed.
            vmcnt_fence<1>();
        } else {
            // Last tile for this block: only the current copy is outstanding.
            vmcnt_fence<0>();
        }
        __builtin_amdgcn_s_barrier();

        out[tile * TILE + tid] = heavy(s[buf][tid], numOps);

        //__builtin_amdgcn_s_barrier();
        buf = nbuf;
    }
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

    int warpSize = 0;
    CHECK_HIP(hipDeviceGetAttribute(&warpSize, hipDeviceAttributeWarpSize, 0));
    if (warpSize != WAVE_SIZE) {
        printf("This sample requires a wavefront size of %d (found %d). Exiting...\n",
               WAVE_SIZE, warpSize);
        return 1;
    }

    int smCount = 0;
    CHECK_HIP(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, 0));
    int numBlocks = smCount * 2;

    printf("Number of elements = %d, Number of tiles = %d, Number of operations per work-item = %d, Number of blocks = %d, iters = %d\n\n",
           size, TILE, numOps, numBlocks, iters);

    float *h_in, *d_in, *d_out, *h_outSync, *h_outPipe;
    CHECK_HIP(hipHostMalloc(&h_in, sizeof(float) * size));
    CHECK_HIP(hipHostMalloc(&h_outSync, sizeof(float) * size));
    CHECK_HIP(hipHostMalloc(&h_outPipe, sizeof(float) * size));
    CHECK_HIP(hipMalloc(&d_in, sizeof(float) * size));
    CHECK_HIP(hipMalloc(&d_out, sizeof(float) * size));

    for (int i = 0; i < size; i++) {
        h_in[i] = (float)((i % 97) * 0.01f + 1.0f);
    }
    CHECK_HIP(hipMemcpy(d_in, h_in, sizeof(float) * size, hipMemcpyHostToDevice));

    for (int it = 0; it < iters; it++) {
        tileSync<<<numBlocks, TILE>>>(d_in, d_out, numTiles, numOps); // warmup
    }
    CHECK_HIP(hipMemcpy(h_outSync, d_out, sizeof(float) * size, hipMemcpyDeviceToHost));

    CHECK_HIP(hipDeviceSynchronize());
    auto start = std::chrono::steady_clock::now();

    for (int it = 0; it < iters; it++) {
        tileSync<<<numBlocks, TILE>>>(d_in, d_out, numTiles, numOps);
    }
    CHECK_HIP(hipDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    double syncTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    syncTime = syncTime / iters;

    for (int it = 0; it < iters; it++) {
        tilePipelined<<<numBlocks, TILE>>>(d_in, d_out, numTiles, numOps); // warmup
    }
    CHECK_HIP(hipMemcpy(h_outPipe, d_out, sizeof(float) * size, hipMemcpyDeviceToHost));

    CHECK_HIP(hipDeviceSynchronize());
    start = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; it++) {
        tilePipelined<<<numBlocks, TILE>>>(d_in, d_out, numTiles, numOps);
    }
    CHECK_HIP(hipDeviceSynchronize());
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
    printf("  Pipelined (async LDS + vmcnt)    : %.4lf ms\n", pipeTime * 1e-6);
    printf("  Speedup (sync / pipelined)       : %.3lfx\n", syncTime / pipeTime);

    CHECK_HIP(hipFree(d_in));
    CHECK_HIP(hipFree(d_out));
    CHECK_HIP(hipHostFree(h_in));
    CHECK_HIP(hipHostFree(h_outSync));
    CHECK_HIP(hipHostFree(h_outPipe));

    bool ok = (mismatches == 0);
    printf("\n%s\n", ok ? "PASS" : "FAIL");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
