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

  const size_t num_Apoints_bytes = sizeof(float2) * num_Apoints;
  const size_t num_Bpoints_bytes = sizeof(float2) * num_Bpoints;

  float2 *h_Apoints = (float2*) malloc (num_Apoints_bytes);
  float2 *h_Bpoints = (float2*) malloc (num_Bpoints_bytes);
  
  srand(123);
  for (int i = 0; i < num_Apoints; i++) {
    h_Apoints[i].x() = (float)rand() / (float)RAND_MAX;
    h_Apoints[i].y() = (float)rand() / (float)RAND_MAX;
  }
  
  for (int i = 0; i < num_Bpoints; i++) {
    h_Bpoints[i].x() = (float)rand() / (float)RAND_MAX;
    h_Bpoints[i].y() = (float)rand() / (float)RAND_MAX;
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float2 *d_Apoints = sycl::malloc_device<float2>(num_Apoints, q);
  q.memcpy(d_Apoints, h_Apoints, num_Apoints_bytes);

  float2 *d_Bpoints = sycl::malloc_device<float2>(num_Bpoints, q);
  q.memcpy(d_Bpoints, h_Bpoints, num_Bpoints_bytes);

  float *d_distance = sycl::malloc_device<float>(2, q);

  sycl::range<1> gwsA ((num_Apoints + 255) / 256 * 256);
  sycl::range<1> gwsB ((num_Bpoints + 255) / 256 * 256);
  sycl::range<1> lws (256);

  float h_distance[2] = {-1.f, -1.f}; 

  double time = 0.0;

  for (int i = 0; i < repeat; i++) {
    q.memcpy(d_distance, h_distance, 2 * sizeof(float)).wait();

    auto start = std::chrono::steady_clock::now();

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class distanceAB>(
        sycl::nd_range<1>(gwsA, lws), [=] (sycl::nd_item<1> item) {
        computeDistance(item, d_Apoints, d_Bpoints, d_distance,
                        num_Apoints, num_Bpoints);
      });
    });

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class distanceBA>(
        sycl::nd_range<1>(gwsB, lws), [=] (sycl::nd_item<1> item) {
        computeDistance(item, d_Bpoints, d_Apoints, d_distance + 1,
                        num_Bpoints, num_Apoints);
      });
    });

    q.wait();
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }
  printf("Average execution time of kernels: %f (ms)\n", (time * 1e-6f) / repeat);

  q.memcpy(h_distance, d_distance, 2 * sizeof(float)).wait();

  printf("Verifying the result may take a while..\n");
  float r_distance = hausdorff_distance(h_Apoints, h_Bpoints, num_Apoints, num_Bpoints);
  float t_distance = std::max(h_distance[0], h_distance[1]);

  bool error = (fabsf(t_distance - r_distance)) > 1e-3f;
  printf("%s\n", error ? "FAIL" : "PASS");
  if (error) exit(1);

  free(h_Apoints);
  free(h_Bpoints);
  sycl::free(d_distance, q);
  sycl::free(d_Apoints, q);
  sycl::free(d_Bpoints, q);
  return 0;
}
