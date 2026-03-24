#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <sycl/sycl.hpp>
#include "reference.h"

#define GPU_NUM_THREADS 256

template <typename scalar_t, typename accscalar_t>
void sampleMultinomialOnce(
    sycl::nd_item<1> &item,
    accscalar_t *smem,
    bool &found,
    int &foundPos,
    int *__restrict dest,
    int distributions,
    int categories,
    const scalar_t*__restrict sampled,
    const scalar_t*__restrict dist,
    int stride_dist,
    int stride_categories)
{

  accscalar_t accZero = static_cast<accscalar_t>(0);
  scalar_t zero = static_cast<scalar_t>(0);

  int threadIdx_x = item.get_local_id(0);
  int blockIdx_x = item.get_group(0);
  int blockDim_x = item.get_local_range(0);
  int gridDim_x = item.get_group_range(0);

  auto g = item.get_group();

  for (int curDist = blockIdx_x; curDist < distributions; curDist += gridDim_x) {
    // Each block handles one distribution
    // First pass, find the total sum of the distribution
    accscalar_t sum = accZero;
    scalar_t val;
    for (int cat = threadIdx_x; cat < categories; cat += blockDim_x) {
      val = dist[curDist * stride_dist + cat * stride_categories];
      sum += static_cast<accscalar_t>(val);
    }

    // threadIdx_x == 0 has the sum value from this
    sum = reduce_over_group(g, sum, std::plus<>());

    // Broadcast sum and sample value
    if (threadIdx_x == 0) {
      // Make sure the sum of our distribution didn't overflow
      foundPos = 0;
      smem[0] = sum;
      smem[1] = sampled[curDist];
    }
    group_barrier(g, sycl::memory_scope::work_group);

    sum = smem[0];
    scalar_t sample = static_cast<scalar_t>(smem[1]);
    group_barrier(g, sycl::memory_scope::work_group);

    // zero sum
    if (sum == accZero) {
      // Choose the first element
      if (threadIdx_x == 0) {
        dest[curDist] = 0;
      }
      continue;
    }

    int chunks = (categories + (int)blockDim_x - 1) / blockDim_x;
    accscalar_t prevHighProb = accZero;
    found = false;

    for (int chunk = 0; chunk < chunks && !found; ++chunk) {
      // All threads in bounds load a value
      int cat = chunk * blockDim_x + threadIdx_x;

      accscalar_t dist_val = cat < categories ?
                             static_cast<accscalar_t>(dist[curDist * stride_dist + cat * stride_categories]) / sum :
                             accZero;

      smem[threadIdx_x] = dist_val;
      group_barrier(g, sycl::memory_scope::work_group);

      // Perform an inclusive prefix sum of the shared memory contents
      for (int offset = 1; offset < blockDim_x; offset *= 2) {
        accscalar_t val = accZero;

        if (threadIdx_x >= offset) {
          val = smem[threadIdx_x - offset] + smem[threadIdx_x];
        }

        group_barrier(g, sycl::memory_scope::work_group);
        if (threadIdx_x >= offset) {
          smem[threadIdx_x] = val;
        }
        group_barrier(g, sycl::memory_scope::work_group);
      }

      // Each thread will check to see if the sample falls in its bucket
      scalar_t curBucket =
          static_cast<scalar_t>(smem[threadIdx_x] + prevHighProb);
      scalar_t prevBucket = static_cast<scalar_t>(
          threadIdx_x == 0 ? prevHighProb : smem[threadIdx_x - 1] + prevHighProb);
      bool inBucket =
          (cat < categories) &&
          (!(sample >= curBucket) &&
          (sample >= prevBucket) &&
          (dist_val > zero));

      if (inBucket) {
        auto ao = sycl::atomic_ref<int, sycl::memory_order::relaxed,
                                   sycl::memory_scope::work_group,
                                   sycl::access::address_space::local_space>(foundPos);
        ao.fetch_max(cat);

        found = true;
      }

      // Store the previous scan's high value for future use
      prevHighProb = prevHighProb + smem[blockDim_x - 1];

      group_barrier(g, sycl::memory_scope::work_group);
    }

    if (threadIdx_x == 0) {
      if (found) {
          dest[curDist] = foundPos;
      } else {
        // This should address a rare bug where we don't select a valid index. This likely occurs when
        // due to floating point arithmetic rounding errors, our cumulative sum does not add up to 1, but
        // and our uniform sample is greater than this value. In this case we likely have unitialized memory
        // in dest[curDist]. So basically we will loop through the distribution and pick the largest index
        // where the distribution is non-zero. This is obviously terribly inefficient, but due to the
        // rarity in which this occurs, this should not be an issue.
        for (int cat = categories - 1; cat >= 0; --cat) {
          if (dist[curDist * stride_dist + cat * stride_categories] > zero) {
            dest[curDist] = cat;
            break;
          }
        }
      }
    }
  }
}


int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <number of distributions> <number of categories> <repeat>\n", argv[0]);
    return 1;
  }
  const int numDist = atoi(argv[1]);
  const int numCategories = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  int sample_size_bytes = numDist * sizeof(float);
  float *sample = (float*) malloc (sample_size_bytes);

  std::default_random_engine g (123);
  std::uniform_real_distribution<float> uniform_distr (0.f, 1.f);

  for (int i = 0; i < numDist; i++) {
    sample[i] = uniform_distr(g);
  }

  int result_size_bytes = numDist * sizeof(int);
  int *result = (int*) malloc (result_size_bytes);
  int *result_ref = (int*) malloc (result_size_bytes);

  size_t distr_size_bytes = numDist * numCategories * sizeof(float);
  float *distr = (float*) malloc (distr_size_bytes);

  srand(123);
  for (int i = 0; i < numDist; i++) {
    for (int j = 0; j < numCategories; j++) {
      // they don't need to sum to 1 in which case the values are weights
      distr[i * numCategories + j] = rand() % 100 + 1;
    }
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_sample = sycl::malloc_device<float>(numDist, q);
  q.memcpy(d_sample, sample, sample_size_bytes);

  float *d_distr = sycl::malloc_device<float>(numDist * numCategories, q);
  q.memcpy(d_distr, distr, distr_size_bytes);

  int *d_result = sycl::malloc_device<int>(numDist, q);

  int requiredThreads = GPU_NUM_THREADS;
  sycl::range<1> gws (512 * requiredThreads);
  sycl::range<1> lws (requiredThreads);

  auto kFn = [&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> smem (sycl::range<1>(requiredThreads), cgh);
    sycl::local_accessor<bool, 0> found (cgh);
    sycl::local_accessor<int, 0> foundPos (cgh);
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      sampleMultinomialOnce<float, float> (
        item, smem.get_multi_ptr<sycl::access::decorated::no>().get(), found, foundPos,
        d_result, numDist, numCategories, d_sample, d_distr, numCategories, 1);
    });
  };

  // warmup and verify
  q.submit(kFn);

  sampleMultinomialOnce_cpu<float, float> (
      result_ref, numDist, numCategories, sample, distr, numCategories, 1);

  q.memcpy(result, d_result, result_size_bytes).wait();

  int error = 0;
  for (int i = 0; i < numDist; i++) {
    if (abs(result[i] - result_ref[i]) > 1) {
      printf("results mismatch: %d %d %d\n", i, result[i], result_ref[i]);
      error = 1;
      break;
    }
  }
  printf("%s\n", error ? "FAIL" : "PASS");

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit(kFn);
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of sampleMultinomialOnce kernel: %f (us)\n", (time * 1e-3f) / repeat);

  sycl::free(d_result, q);
  sycl::free(d_sample, q);
  sycl::free(d_distr, q);

  free(result_ref);
  free(result);
  free(sample);
  free(distr);

  return 0;
}
