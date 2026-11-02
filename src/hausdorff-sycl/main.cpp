#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <sycl/sycl.hpp>

#ifdef __NVPTX__
  #include <sycl/ext/oneapi/experimental/cuda/builtins.hpp>
  using namespace sycl::ext::oneapi::experimental::cuda;
#else
  #define ldg(a) (*(a))
#endif

using float2 = sycl::float2;

#include "reference.h"

inline float hd (const float2 ap, const float2 bp)
{
  return (ap.x() - bp.x()) * (ap.x() - bp.x())
       + (ap.y() - bp.y()) * (ap.y() - bp.y());
}

void computeDistance(sycl::nd_item<1> &item,
                     const float2* __restrict Apoints,
                     const float2* __restrict Bpoints,
                           float*  __restrict distance,
                     const int numA, const int numB)
{
  int i = item.get_global_id(0);
  if (i >= numA) return;

  float d = std::numeric_limits<float>::max();
  float2 p = ldg(&Apoints[i]);
  for (int j = 0; j < numB; j++)
  {
    float t = hd(p, ldg(&Bpoints[j]));
    d = sycl::min(t, d);
  }
  
  auto ao = sycl::atomic_ref<float, 
            sycl::memory_order::relaxed,
            sycl::memory_scope::device,
            sycl::access::address_space::global_space> (distance[0]);
  ao.fetch_max(d);
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <number of points in space A>", argv[0]);
    printf(" <number of points in space B> <repeat>\n");
    return 1;
  }
  const int num_Apoints = atoi(argv[1]);
  const int num_Bpoints = atoi(argv[2]);
  const int repeat = atoi(argv[3]);
  if (repeat < 2) {
    printf("repeat must be at least 2\n");
    return 1;
  }

  const size_t num_Apoints_bytes = sizeof(float2) * num_Apoints;
  const size_t num_Bpoints_bytes = sizeof(float2) * num_Bpoints;

  float2 *h_A[2], *h_B[2];
  for (int b = 0; b < 2; b++) {
    h_A[b] = (float2*) malloc(num_Apoints_bytes);
    h_B[b] = (float2*) malloc(num_Bpoints_bytes);
  }

  srand(123);
  for (int b = 0; b < 2; b++) {
    for (int i = 0; i < num_Apoints; i++) {
      h_A[b][i].x() = (float)rand() / (float)RAND_MAX;
      h_A[b][i].y() = (float)rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < num_Bpoints; i++) {
      h_B[b][i].x() = (float)rand() / (float)RAND_MAX;
      h_B[b][i].y() = (float)rand() / (float)RAND_MAX;
      // Slot 0 is shifted so its distance stays above slot 1 for any point
      // count. Iteration 1 then keeps slot 0 whenever -1 was uploaded once.
      if (b == 0) {
        h_B[b][i].x() += 10.f;
        h_B[b][i].y() += 10.f;
      }
    }
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float2 *d_A[2], *d_B[2];
  for (int b = 0; b < 2; b++) {
    d_A[b] = sycl::malloc_device<float2>(num_Apoints, q);
    q.memcpy(d_A[b], h_A[b], num_Apoints_bytes);
    d_B[b] = sycl::malloc_device<float2>(num_Bpoints, q);
    q.memcpy(d_B[b], h_B[b], num_Bpoints_bytes);
  }

  float *d_distance = sycl::malloc_device<float>(2, q);

  sycl::range<1> gwsA ((num_Apoints + 255) / 256 * 256);
  sycl::range<1> gwsB ((num_Bpoints + 255) / 256 * 256);
  sycl::range<1> lws (256);

  // The atomic max keeps the value already at this address. A single
  // upload of -1 therefore leaves the larger slot in place after iteration 0.
  float h_distance[2] = {-1.f, -1.f};
  float got[2];

  double time = 0.0;

  for (int i = 0; i < repeat; i++) {
    const int slot = i & 1;

    h_distance[0] = h_distance[1] = -1.f;
    q.memcpy(d_distance, h_distance, 2 * sizeof(float)).wait();

    const float2 *A = d_A[slot];
    const float2 *B = d_B[slot];

    auto start = std::chrono::steady_clock::now();

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class distanceAB>(
        sycl::nd_range<1>(gwsA, lws), [=] (sycl::nd_item<1> item) {
        computeDistance(item, A, B, d_distance,
                        num_Apoints, num_Bpoints);
      });
    });

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class distanceBA>(
        sycl::nd_range<1>(gwsB, lws), [=] (sycl::nd_item<1> item) {
        computeDistance(item, B, A, d_distance + 1,
                        num_Bpoints, num_Apoints);
      });
    });

    q.wait();
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

    // The two slots are fully covered by the first two iterations.
    if (i < 2) {
      q.memcpy(h_distance, d_distance, 2 * sizeof(float)).wait();
      got[i] = std::max(h_distance[0], h_distance[1]);
    }
  }
  printf("Average execution time of kernels: %f (ms)\n", (time * 1e-6f) / repeat);

  printf("Verifying the result may take a while..\n");
  bool error = false;
  for (int s = 0; s < 2; s++) {
    float r_distance = hausdorff_distance(h_A[s], h_B[s], num_Apoints, num_Bpoints);
    if (fabsf(got[s] - r_distance) > 1e-3f * r_distance)
      error = true;
  }
  printf("%s\n", error ? "FAIL" : "PASS");

  for (int b = 0; b < 2; b++) {
    free(h_A[b]);
    free(h_B[b]);
    sycl::free(d_A[b], q);
    sycl::free(d_B[b], q);
  }
  sycl::free(d_distance, q);
  return 0;
}
