#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <sycl/sycl.hpp>
#include "reference.h"

//
// Assumption
// There are many more evaluation(target) points than sources for the subsequent code.
// Each thread block will perform the evaluation for a small chunk of the target points and all source points.
//
void matern_kernel (
  sycl::nd_item<2> &item,
  const int num_targets,
  const float l,
  const float *__restrict sources,
  const float *__restrict targets,
  const float *__restrict weights,
        float *__restrict result,
        float *__restrict local_result,
        float *__restrict local_targets,
        float *__restrict local_sources,
        float *__restrict local_weights)

{
  int tx = item.get_local_id(1);
  int px = item.get_global_id(1); // range [0:ntargets)
  if (px >= num_targets) return;

  int ty = item.get_local_id(0);
  int py = ty; // range [0:nsources)
  if (py >= SY) return;

  if (ty == 0) {
    for (int i = 0; i < 3; i++)
      local_targets[tx * 3 + i] = targets[px * 3 + i];
  }

  if (tx == 0) {
    for (int i = 0; i < 3; i++)
      local_sources[ty * 3 + i] = sources[py * 3 + i];
    local_weights[ty] = weights[ty];
  }

  item.barrier(sycl::access::fence_space::local_space);

  float squared_diff = 0.f;

  for (int i = 0; i < 3; i++) {
    squared_diff += (local_targets[tx * 3 + i] - local_sources[ty * 3 + i]) *
                    (local_targets[tx * 3 + i] - local_sources[ty * 3 + i]);
  }
  float diff = sycl::sqrt(squared_diff);

  local_result[tx * SY + ty] =
    (1.f + sycl::sqrt(5.f) * diff / l + 5.f * squared_diff / (3.f * l * l)) *
    sycl::exp(-sycl::sqrt(5.f) * diff / l) * local_weights[ty];

  item.barrier(sycl::access::fence_space::local_space);

  if (ty == 0) {
    float res = 0.f;
    for (int i = 0; i < SY; i++)
      res += local_result[tx * SY + i];
    result[px] = res;
  }
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of points> <repeat>\n", argv[0]);
    return 1;
  }
  const int npoints = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  const int source_size = nsources * 3;  // (x,y,z) coordinates in a 3D grid
  const int source_size_byte = source_size * sizeof(float);

  const int weight_size = nsources;
  const int weight_size_byte = weight_size * sizeof(float);

  const int ntargets = npoints * npoints * npoints;
  const int target_size = ntargets * 3;
  const int target_size_byte = target_size * sizeof(float);

  const int result_size = ntargets;
  const int result_size_byte = ntargets * sizeof(float);

  float *sources = (float*) malloc (source_size_byte);
  float *targets = (float*) malloc (target_size_byte);
  float *weights = (float*) malloc (weight_size_byte);
  float *result = (float*) malloc (result_size_byte);
  float *result_ref = (float*) malloc (result_size_byte);

  srand(123);
  for (int i = 0; i < source_size; i++)
    sources[i] = rand() / (float)RAND_MAX;

  for (int i = 0; i < weight_size; i++)
    weights[i] = rand() / (float)RAND_MAX;

  for (int i = 0; i < target_size; i++)
    targets[i] = rand() / (float)RAND_MAX;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_sources = sycl::malloc_device<float>(source_size, q);
  float *d_weights = sycl::malloc_device<float>(weight_size, q);
  float *d_targets = sycl::malloc_device<float>(target_size, q);
  float *d_result = sycl::malloc_device<float>(result_size, q);

  q.memcpy(d_sources, sources, source_size_byte);
  q.memcpy(d_weights, weights, weight_size_byte);
  q.memcpy(d_targets, targets, target_size_byte);
  q.wait();

  float l = 0.1f; // length scale lower bound

  const int nblocks = (ntargets + SX - 1) / SX;
  sycl::range<2> gws (64, SX * nblocks);
  sycl::range<2> lws (64, SX);

  // quickly verify the results using a small problem size
  const int ntargets_small = 16*16*16;
  printf("------------------------------------------------------------\n");
  printf("Verifying the kernel results with the problem size (16 cube)\n");
  printf("------------------------------------------------------------\n");

  while (l <= 1e5f) {
    auto e = q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<float, 1> l_result (sycl::range<1>(SX*SY), cgh);
      sycl::local_accessor<float, 1> l_targets (sycl::range<1>(SX*3), cgh);
      sycl::local_accessor<float, 1> l_sources (sycl::range<1>(SY*3), cgh);
      sycl::local_accessor<float, 1> l_weights (sycl::range<1>(SY), cgh);
      cgh.parallel_for<class test>(
        sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
        matern_kernel(item, ntargets_small, l, d_sources, d_targets, d_weights, d_result,
                      l_result.get_multi_ptr<sycl::access::decorated::no>().get(),
                      l_targets.get_multi_ptr<sycl::access::decorated::no>().get(),
                      l_sources.get_multi_ptr<sycl::access::decorated::no>().get(),
                      l_weights.get_multi_ptr<sycl::access::decorated::no>().get());
      });
    });

    matern_kernel_reference(nsources, ntargets_small, l, sources, targets, weights, result_ref);

    q.memcpy(result, d_result, ntargets_small * sizeof(float), e).wait();

    bool ok = true;
    for (int i = 0; i < ntargets_small; i++) {
      if (fabsf(result[i] - result_ref[i]) > 1e-3f) {
        printf("@%d actual=%f expected=%f\n", i, result[i] , result_ref[i]);
        ok = false;
        break;
      }
    }
    printf("Length scale = %.1e check = %s\n", l, ok ? "PASS" : "FAIL");
    if (!ok) exit(1);
    l = l * 10.f;
  }

  printf("--------------------------------------------------------------------\n");
  printf("Timing the kernel execution with the problem size (%d cube)\n", npoints);
  printf("--------------------------------------------------------------------\n");

  l = 0.1f;
  while (l <= 1e5f) {
    printf("Warmup..\n");
    for (int i = 0; i < repeat; i++) {
      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<float, 1> l_result (sycl::range<1>(SX*SY), cgh);
        sycl::local_accessor<float, 1> l_targets (sycl::range<1>(SX*3), cgh);
        sycl::local_accessor<float, 1> l_sources (sycl::range<1>(SY*3), cgh);
        sycl::local_accessor<float, 1> l_weights (sycl::range<1>(SY), cgh);
        cgh.parallel_for<class warmup>(
          sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
          matern_kernel(item, ntargets, l, d_sources, d_targets, d_weights, d_result,
                        l_result.get_multi_ptr<sycl::access::decorated::no>().get(),
                        l_targets.get_multi_ptr<sycl::access::decorated::no>().get(),
                        l_sources.get_multi_ptr<sycl::access::decorated::no>().get(),
                        l_weights.get_multi_ptr<sycl::access::decorated::no>().get());
        });
      });
    }
    q.wait();

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<float, 1> l_result (sycl::range<1>(SX*SY), cgh);
        sycl::local_accessor<float, 1> l_targets (sycl::range<1>(SX*3), cgh);
        sycl::local_accessor<float, 1> l_sources (sycl::range<1>(SY*3), cgh);
        sycl::local_accessor<float, 1> l_weights (sycl::range<1>(SY), cgh);
        cgh.parallel_for<class measure>(
          sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
          matern_kernel(item, ntargets, l, d_sources, d_targets, d_weights, d_result,
                        l_result.get_multi_ptr<sycl::access::decorated::no>().get(),
                        l_targets.get_multi_ptr<sycl::access::decorated::no>().get(),
                        l_sources.get_multi_ptr<sycl::access::decorated::no>().get(),
                        l_weights.get_multi_ptr<sycl::access::decorated::no>().get());
        });
      });
    }

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Length scale = %.1e ", l);
    printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

    l = l * 10.f;
  }

  sycl::free(d_sources, q);
  sycl::free(d_weights, q);
  sycl::free(d_targets, q);
  sycl::free(d_result, q);

  free(sources);
  free(weights);
  free(targets);
  free(result);
  free(result_ref);
  return 0;
}
