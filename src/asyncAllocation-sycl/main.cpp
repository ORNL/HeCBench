/*
 * Asynchronous (stream-ordered) memory allocation benchmark.
 *
 * Real use case modelled here: a streaming / batched inference-style service
 * that receives a queue of independent "jobs" of *varying* problem sizes and
 * processes them concurrently over a small pool of SYCL queues.  Each job
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
 *   1. Synchronous allocation  (sycl::malloc_device / sycl::free)
 *      sycl::free requires that the memory no longer be in use, so a queue
 *      synchronization is needed before every free.  That serializes the
 *      queues and pays the driver allocation cost on every job.
 *
 *   2. Asynchronous allocation (async_malloc / async_free)
 *      Allocations are stream-ordered on an in-order queue and served from a
 *      memory pool.  Freed memory is recycled across jobs, so after warm-up
 *      there is no driver round-trip and no host-side synchronization -> the
 *      queues overlap.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include <vector>
#include <random>
#include <iostream>
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/async_alloc/async_alloc.hpp>
#include <sycl/ext/oneapi/experimental/async_alloc/memory_pool.hpp>

namespace syclexp = sycl::ext::oneapi::experimental;

#define NSTREAMS 4

// Host reference used to spot-check a few elements.
static float process_ref(float v, int iter) {
  for (int k = 0; k < iter; k++) {
    v = v * 0.9999f + 0.0001f;
    v = sinf(v) * cosf(v) + sqrtf(fabsf(v) + 1.0f);
  }
  return v;
}

// Synchronous allocation: sycl::malloc_device / sycl::free per job.
double run_sync(const float *h_in, float *h_out,
                const std::vector<size_t> &sizes,
                const std::vector<size_t> &offsets,
                sycl::queue *queues, int iter) {
  auto start = std::chrono::steady_clock::now();
  for (size_t j = 0; j < sizes.size(); j++) {
    const size_t n = sizes[j];
    const size_t bytes = n * sizeof(float);
    sycl::queue &q = queues[j % NSTREAMS];

    float *d_in = sycl::malloc_device<float>(n, q);
    float *d_out = sycl::malloc_device<float>(n, q);

    q.memcpy(d_in, h_in + offsets[j], bytes);
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
      float v = d_in[idx];
      for (int k = 0; k < iter; k++) {
        v = v * 0.9999f + 0.0001f;
        v = sycl::sin(v) * sycl::cos(v) + sycl::sqrt(sycl::fabs(v) + 1.0f);
      }
      d_out[idx] = v;
    });
    q.memcpy(h_out + offsets[j], d_out, bytes);

    // sycl::free requires the memory to no longer be in use -> synchronize.
    q.wait();
    sycl::free(d_in, q);
    sycl::free(d_out, q);
  }
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

// Asynchronous (stream-ordered) allocation: async_malloc_from_pool / async_free.
//
// An explicit memory pool with a non-zero reserve threshold is used so that
// freed memory is retained by the pool (up to the threshold) and recycled by
// subsequent allocations, instead of being released back to the memory
// provider on every free.  This is the property the extension relies on for
// the pool to deliver its performance benefit (see the extension overview and
// the "reserve threshold" examples in the spec).
//
// NOTE: the reserve threshold is a *hint*; some backends currently ignore it
// (e.g. the Level Zero adapter hard-wires the threshold to 0 today), in which
// case this simply behaves like the default pool.
double run_async(const float *h_in, float *h_out,
                 const std::vector<size_t> &sizes,
                 const std::vector<size_t> &offsets,
                 sycl::queue *queues, int iter,
                 const syclexp::memory_pool &pool) {
  auto start = std::chrono::steady_clock::now();
  for (size_t j = 0; j < sizes.size(); j++) {
    const size_t n = sizes[j];
    const size_t bytes = n * sizeof(float);
    sycl::queue &q = queues[j % NSTREAMS];

    // In-order queue guarantees alloc -> copy -> kernel -> copy -> free order.
    float *d_in = (float *)syclexp::async_malloc_from_pool(q, bytes, pool);
    float *d_out = (float *)syclexp::async_malloc_from_pool(q, bytes, pool);

    q.memcpy(d_in, h_in + offsets[j], bytes);
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
      float v = d_in[idx];
      for (int k = 0; k < iter; k++) {
        v = v * 0.9999f + 0.0001f;
        v = sycl::sin(v) * sycl::cos(v) + sycl::sqrt(sycl::fabs(v) + 1.0f);
      }
      d_out[idx] = v;
    });
    q.memcpy(h_out + offsets[j], d_out, bytes);

    syclexp::async_free(q, d_in);
    syclexp::async_free(q, d_out);
  }
  for (int i = 0; i < NSTREAMS; i++) queues[i].wait();
  auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

int main(int argc, char **argv) {
  int num_jobs = (argc > 1) ? atoi(argv[1]) : 2000;
  int iter     = (argc > 2) ? atoi(argv[2]) : 50;

#ifdef USE_GPU
  sycl::queue probe(sycl::gpu_selector_v);
#else
  sycl::queue probe(sycl::cpu_selector_v);
#endif
  if (!probe.get_device().has(sycl::aspect::ext_oneapi_async_memory_alloc)) {
    printf("Skip execution as device does not support async memory allocation\n");
    return 1;
  }

  // A shared context so all queues target the same device / memory pool.
  sycl::device dev = probe.get_device();
  sycl::context ctx = probe.get_context();
  sycl::property_list props{sycl::property::queue::in_order()};
  sycl::queue queues[NSTREAMS] = {
      sycl::queue(ctx, dev, props), sycl::queue(ctx, dev, props),
      sycl::queue(ctx, dev, props), sycl::queue(ctx, dev, props)};

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

  printf("Streaming pipeline: %d jobs over %d queues, sizes [%zu, %zu], "
         "%d compute iterations/element\n",
         num_jobs, NSTREAMS, minN, maxN, iter);
  printf("Total working set: %.1f MB per array\n",
         total * sizeof(float) / (1024.0 * 1024.0));

  // Explicit device memory pool with a non-zero reserve threshold.  The
  // threshold instructs the runtime to keep this much freed memory in the pool
  // (rather than returning it to the provider) so it can be recycled by later
  // allocations.  We size it to comfortably hold the buffers that can be live
  // and/or freed-but-cached across the small set of concurrent queues.
  const size_t pool_threshold = 2 * NSTREAMS * maxN * sizeof(float) * 8;
  syclexp::memory_pool pool(
      ctx, dev, sycl::usm::alloc::device,
      syclexp::properties{syclexp::initial_threshold{pool_threshold}});
  // Request the threshold via the setter, in case a backend honors one path but not the other.
  pool.increase_threshold_to(pool_threshold);
  printf("Pool reserve threshold requested: %.1f MB (reported: %.1f MB)\n",
         pool_threshold / (1024.0 * 1024.0),
         pool.get_threshold() / (1024.0 * 1024.0));

  float *h_in = sycl::malloc_host<float>(total, ctx);
  float *h_out_sync = sycl::malloc_host<float>(total, ctx);
  float *h_out_async = sycl::malloc_host<float>(total, ctx);

  for (int j = 0; j < num_jobs; j++)
    for (size_t i = 0; i < sizes[j]; i++)
      h_in[offsets[j] + i] = sinf((float)j * 0.001f + (float)i * 1e-6f);

  // Warm up both paths (JIT + pool steady state), untimed.
  run_async(h_in, h_out_async, sizes, offsets, queues, iter, pool);
  run_sync(h_in, h_out_sync, sizes, offsets, queues, iter);

  double t_async = run_async(h_in, h_out_async, sizes, offsets, queues, iter, pool);
  double t_sync  = run_sync(h_in, h_out_sync, sizes, offsets, queues, iter);

  printf("Pool reserved/used after run: %.1f / %.1f MB\n",
         pool.get_reserved_size_current() / (1024.0 * 1024.0),
         pool.get_used_size_current() / (1024.0 * 1024.0));
  printf("\nSynchronous  (malloc_device / free)  : %8.2f ms\n", t_sync);
  printf("Asynchronous (async_malloc / async_free): %8.2f ms\n", t_async);
  printf("Speedup from asynchronous allocation : %8.2fx\n", t_sync / t_async);

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

  sycl::free(h_in, ctx);
  sycl::free(h_out_sync, ctx);
  sycl::free(h_out_async, ctx);

  return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
