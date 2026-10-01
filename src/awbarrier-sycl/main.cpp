// A software-pipelined (double-buffered) kernel that overlaps asynchronous
// global->local memory copies of the NEXT tile with the compute on the CURRENT
// tile.
//
#include <chrono>
#include <cstdio>
#include <cmath>
#include <sycl/sycl.hpp>

#ifndef TILE
#define TILE 256 // elements per tile == work-items per work-group
#endif

// Compute-heavy transform applied to each element after it is staged in local
// memory. Enough work that hiding the global-load latency actually matters.
inline float heavy(float v, int numOps)
{
    for (int k = 0; k < numOps; k++) {
        v = sycl::fma(v, 0.9999997f, 0.0000013f);
        v = sycl::fma(v, 1.0000003f, -0.0000007f);
    }
    return v;
}

// Synchronous version: load tile -> barrier -> compute.
// The load latency is exposed at the barrier on every tile.
void tileSync(const sycl::nd_item<1> &item,
              const float *in, float *out, int numTiles, int numOps,
              float *s)
{
    const int tid      = item.get_local_id(0);
    const int blockIdx = item.get_group(0);
    const int gridDim  = item.get_group_range(0);

    for (int tile = blockIdx; tile < numTiles; tile += gridDim) {
        int gidx = tile * TILE + tid;

        s[tid] = in[gidx];
        item.barrier(sycl::access::fence_space::local_space);

        out[gidx] = heavy(s[tid], numOps);
        item.barrier(sycl::access::fence_space::local_space);
    }
}

// Pipelined version: prefetch the next tile with async_work_group_copy while
// computing the current tile. Two local buffers implement the double buffer,
// and a per-buffer sycl::device_event tracks the in-flight copies: we wait only
// for the CURRENT tile's copy, so the NEXT tile's copy keeps running during the
// compute.
void tilePipelined(const sycl::nd_item<1> &item,
                   float *in, float *out, int numTiles, int numOps,
                   float *s /* 2 * TILE */)
{
    using sycl::access::address_space;
    using sycl::access::decorated;

    auto grp           = item.get_group();
    const int tid      = item.get_local_id(0);
    const int blockIdx = item.get_group(0);
    const int gridDim  = item.get_group_range(0);

    auto ldst = [&](int buf) {
        return sycl::address_space_cast<address_space::local_space, decorated::yes>(
            &s[buf * TILE]);
    };
    auto gsrc = [&](int tile) {
        return sycl::address_space_cast<address_space::global_space, decorated::yes>(
            &in[tile * TILE]);
    };

    // Nothing to do for groups whose first tile is out of range.
    int firstTile = blockIdx;
    if (firstTile >= numTiles) return;

    // Async copy for this group's first tile into buffer 0.
    // sycl::device_event is not default-constructible, so we carry the
    // "current" tile's event in a single object instead of an array.
    sycl::device_event cur = grp.async_work_group_copy(ldst(0), gsrc(firstTile), (size_t)TILE);

    int buf = 0;
    for (int tile = blockIdx; tile < numTiles; tile += gridDim) {
        int nextTile = tile + gridDim;
        int nbuf     = buf ^ 1;
        bool hasNext = nextTile < numTiles;

        // Start loading the next tile into the other buffer (kept running in
        // the background). When there is no next tile, reuse cur as a harmless
        // placeholder; it is never waited on again.
        sycl::device_event next =
            hasNext ? grp.async_work_group_copy(ldst(nbuf), gsrc(nextTile), (size_t)TILE)
                    : cur;

        // Wait only for the current tile's copy; the next tile's copy is still
        // running in the background during the compute below.
        cur.wait();

        out[tile * TILE + tid] = heavy(s[buf * TILE + tid], numOps);

        // Make sure every work-item is done reading s[buf] before it is reused.
        //item.barrier(sycl::access::fence_space::local_space);

        cur = next;
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

#ifdef USE_GPU
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
    sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

    int cuCount = q.get_device().get_info<sycl::info::device::max_compute_units>();
    int numBlocks = cuCount * 2;

    sycl::range<1> gws (numBlocks * TILE);
    sycl::range<1> lws (TILE);

    printf("Number of elements = %d, Number of tiles = %d, Number of blocks = %d, iters = %d\n\n",
           size, TILE, numBlocks, iters);

    float *h_in      = (float *) malloc(sizeof(float) * size);
    float *h_outSync = (float *) malloc(sizeof(float) * size);
    float *h_outPipe = (float *) malloc(sizeof(float) * size);

    float *d_in  = sycl::malloc_device<float>(size, q);
    float *d_out = sycl::malloc_device<float>(size, q);

    for (int i = 0; i < size; i++) {
        h_in[i] = (float)((i % 97) * 0.01f + 1.0f);
    }
    q.memcpy(d_in, h_in, sizeof(float) * size).wait();

    auto sync_kernel = [&](sycl::handler &cgh) {
      sycl::local_accessor<float, 1> s(sycl::range<1>(TILE), cgh);
      cgh.parallel_for(
        sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
          tileSync(item, d_in, d_out, numTiles, numOps,
                   s.get_multi_ptr<sycl::access::decorated::no>().get());
      });
    };

    for (int it = 0; it < iters; it++) {  // warmup
        q.submit(sync_kernel);
    }
    q.memcpy(h_outSync, d_out, sizeof(float) * size).wait();

    auto start = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; it++) {
        q.submit(sync_kernel);
    }
    q.wait();
    auto end = std::chrono::steady_clock::now();
    double syncTime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    syncTime = syncTime / iters;

    auto pipe_kernel = [&](sycl::handler &cgh) {
      sycl::local_accessor<float, 1> s(sycl::range<1>(2 * TILE), cgh);
      cgh.parallel_for<class tile_pipe_warmup>(
        sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
          tilePipelined(item, d_in, d_out, numTiles, numOps,
                        s.get_multi_ptr<sycl::access::decorated::no>().get());
      });
    };

    for (int it = 0; it < iters; it++) {  // warmup
        q.submit(pipe_kernel);
    }
    q.memcpy(h_outPipe, d_out, sizeof(float) * size).wait();

    start = std::chrono::steady_clock::now();
    for (int it = 0; it < iters; it++) {
        q.submit(pipe_kernel);
    }
    q.wait();
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
    printf("  Synchronous (barrier)               : %.4lf ms\n", syncTime * 1e-6);
    printf("  Pipelined (async_work_group_copy)   : %.4lf ms\n", pipeTime * 1e-6);
    printf("  Speedup (sync / pipelined)          : %.3lfx\n", syncTime / pipeTime);

    sycl::free(d_in, q);
    sycl::free(d_out, q);
    free(h_in);
    free(h_outSync);
    free(h_outPipe);

    bool ok = (mismatches == 0);
    printf("\n%s\n", ok ? "PASS" : "FAIL");
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
