#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <sycl/sycl.hpp>

#define syncthreads() item.barrier(sycl::access::fence_space::local_space)
#define threadfence() sycl::atomic_fence(sycl::memory_order::acq_rel,\
                                         sycl::memory_scope::device)

void sum (
    sycl::nd_item<1> &item,
    float &partialSum,
    bool &isLastBlockDone,
    const float*__restrict array,
    const int N,
    unsigned int *__restrict count,
    volatile float*__restrict result)
{

  // Each block sums a subset of the input array.
  unsigned int bid = item.get_group(0);
  unsigned int num_blocks = item.get_group_range(0);
  unsigned int block_size = item.get_local_range(0);
  unsigned int lid = item.get_local_id(0);
  unsigned int gid = bid * block_size + lid;

  auto psum = sycl::atomic_ref<float,
              sycl::memory_order::relaxed,
              sycl::memory_scope::work_group,
              sycl::access::address_space::local_space> (partialSum);

  if (lid == 0) partialSum = 0;
  syncthreads();

  if (gid < N)
    psum.fetch_add(array[gid]);

  syncthreads();

  if (lid == 0) {

    // Thread 0 of each block stores the partial sum
    // to global memory. The compiler will use
    // a store operation that bypasses the L1 cache
    // since the "result" variable is declared as
    // volatile. This ensures that the threads of
    // the last block will read the correct partial
    // sums computed by all other blocks.
    result[bid] = partialSum;

    // Thread 0 makes sure that the incrementation
    // of the "count" variable is only performed after
    // the partial sum has been written to global memory.
    threadfence();

    // Thread 0 signals that it is done.
    auto gmem_write = sycl::atomic_ref<unsigned int,
                      sycl::memory_order::relaxed,
                      sycl::memory_scope::device,
                      sycl::access::address_space::global_space> (*count);
    unsigned int value = gmem_write.fetch_add(1u);

    // Thread 0 determines if its block is the last
    // block to be done.
    isLastBlockDone = (value == (num_blocks - 1));
  }

  // Synchronize to make sure that each thread reads
  // the correct value of isLastBlockDone.
  syncthreads();

  if (isLastBlockDone) {

    // The last block sums the partial sums
    // stored in result[0 .. num_blocks-1]
    if (lid == 0) partialSum = 0;
    syncthreads();

    for (int i = lid; i < num_blocks; i += block_size)
      psum.fetch_add(result[i]);

    syncthreads();

    if (lid == 0) {

      // Thread 0 of last block stores the total sum
      // to global memory and resets the count
      // varialble, so that the next kernel call
      // works properly.
      result[0] = partialSum;
      *count = 0;
    }
  }
}

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("Usage: %s <repeat> <array length>\n", argv[0]);
    return 1;
  }

  const int repeat = atoi(argv[1]);
  const int N = atoi(argv[2]);

  float* h_array = (float*) malloc (N * sizeof(float));
  float h_sum;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  const int blocks = 256;
  const int grids = (N + blocks - 1) / blocks;
  sycl::range<1> gws (grids * blocks);
  sycl::range<1> lws (blocks);

  float *d_result = sycl::malloc_device<float>(grids, q);
  float *d_array = sycl::malloc_device<float>(N, q);
  unsigned int *d_count = sycl::malloc_device<unsigned int>(1, q);

  q.memset(d_count, 0, sizeof(unsigned int));

  bool ok = true;
  double time = 0.0;

  for (int i = 0; i < N; i++) h_array[i] = -1.f;
  q.memcpy(d_array, h_array, N * sizeof(float)).wait();

  for (int n = 0; n < repeat; n++) {
    auto start = std::chrono::steady_clock::now();

    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<float, 0> lsum (cgh);
      sycl::local_accessor<bool, 0> isLastBlockDone (cgh);
      cgh.parallel_for<class reduce>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        sum (item, lsum, isLastBlockDone,
             d_array, N, d_count, d_result);
      });
    }).wait();

    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }

  q.memcpy(&h_sum, d_result, sizeof(float)).wait();

  if (h_sum != -1.f * N) {
    ok = false;
  }

  if (ok) printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);

  free(h_array);
  sycl::free(d_result, q);
  sycl::free(d_array, q);
  sycl::free(d_count, q);

  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);
  return 0;
}
