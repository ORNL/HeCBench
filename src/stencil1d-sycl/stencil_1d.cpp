/*
   Shared memory speeds up performance when we need to access data frequently.
   Here, the 1D stencil kernel adds all its neighboring data within a radius.

   The C model is added to verify the stencil result on a GPU
*/

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <sycl/sycl.hpp>

#define RADIUS 7
#define BLOCK_SIZE 256

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <length> <repeat>\n", argv[0]);
    printf("length is a multiple of %d\n", BLOCK_SIZE);
    return 1;
  }
  const int length = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  const int pad_size = (length + RADIUS);

  const size_t input_size_bytes = pad_size * sizeof(int);
  const size_t output_size_bytes = length * sizeof(int);

  // Alloc space for host copies of a, b, c and setup input values
  int* a = (int *)malloc(input_size_bytes);
  int* b = (int *)malloc(output_size_bytes);

  for (int i = 0; i < pad_size; i++) a[i] = i;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  int *d_in = sycl::malloc_device<int>(pad_size, q);
  q.memcpy(d_in, a, input_size_bytes);

  int *d_out = sycl::malloc_device<int>(length, q);

  sycl::range<1> gws (length);
  sycl::range<1> lws (BLOCK_SIZE);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&](sycl::handler& cgh) {
      sycl::local_accessor <int, 1> temp (sycl::range<1>(BLOCK_SIZE + 2 * RADIUS), cgh);
      cgh.parallel_for<class stencil1D>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        int gindex = item.get_global_id(0);
        int lindex = item.get_local_id(0) + RADIUS;

        // Read input elements into shared memory
        temp[lindex] = d_in[gindex];

        // At both end of a block, the sliding window moves beyond the block boundary.
        if (item.get_local_id(0) < RADIUS) {
          temp[lindex - RADIUS] = (gindex < RADIUS) ? 0 : d_in[gindex - RADIUS];
          temp[lindex + BLOCK_SIZE] = d_in[gindex + BLOCK_SIZE];
        }

        // Synchronize (ensure all the threads will be completed before continue)
        item.barrier(sycl::access::fence_space::local_space);

        // Apply the 1D stencil
        int result = 0;
        for (int offset = -RADIUS ; offset <= RADIUS ; offset++)
          result += temp[lindex + offset];

        // Store the result
        d_out[gindex] = result;

      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  // Copy result back to host
  q.memcpy(b, d_out, output_size_bytes).wait();

  // verification
  bool ok = true;
  for (int i = 0; i < 2*RADIUS; i++) {
    int s = 0;
    for (int j = i; j <= i+2*RADIUS; j++)
      s += j < RADIUS ? 0 : (a[j] - RADIUS);
    if (s != b[i]) {
      printf("Error at %d: %d (host) != %d (device)\n", i, s, b[i]);
      ok = false;
      break;
    }
  }

  for (int i = 2*RADIUS; i < length; i++) {
    int s = 0;
    for (int j = i-RADIUS; j <= i+RADIUS; j++)
      s += a[j];
    if (s != b[i]) {
      printf("Error at %d: %d (host) != %d (device)\n", i, s, b[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  // Cleanup
  free(a);
  free(b);
  sycl::free(d_in, q);
  sycl::free(d_out, q);
  if (!ok) return 1;
  return 0;
}
