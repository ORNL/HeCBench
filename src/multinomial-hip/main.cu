#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include "reference.h"

#define GPU_NUM_THREADS 256

template <typename scalar_t, typename accscalar_t>
__global__ void sampleMultinomialOnce(
    int* dest,
    int distributions,
    int categories,
    const scalar_t*__restrict__ sampled,
    const scalar_t*__restrict__ dist,
    int stride_dist,
    int stride_categories)
{
  __shared__ accscalar_t smem[GPU_NUM_THREADS];
  __shared__ bool found;
  __shared__ int foundPos;

  typedef hipcub::BlockReduce<accscalar_t, GPU_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  accscalar_t accZero = static_cast<accscalar_t>(0);
  scalar_t zero = static_cast<scalar_t>(0);

  for (int curDist = blockIdx.x; curDist < distributions; curDist += gridDim.x) {
    // Each block handles one distribution
    // First pass, find the total sum of the distribution
    accscalar_t sum = accZero;
    scalar_t val;
    for (int cat = threadIdx.x; cat < categories; cat += blockDim.x) {
      val = dist[curDist * stride_dist + cat * stride_categories];
      sum += static_cast<accscalar_t>(val);
    }

    // threadIdx.x == 0 has the sum value from this
    sum = BlockReduce(temp_storage).Sum(sum);

    // Broadcast sum and sample value
    if (threadIdx.x == 0) {
      // Make sure the sum of our distribution didn't overflow
      foundPos = 0;
      smem[0] = sum;
      smem[1] = sampled[curDist];
    }
    __syncthreads();

    sum = smem[0];
    scalar_t sample = static_cast<scalar_t>(smem[1]);
    __syncthreads();

    // zero sum
    if (sum == accZero) {
      // Choose the first element
      if (threadIdx.x == 0) {
        dest[curDist] = 0;
      }
      continue;
    }

    int chunks = (categories + (int)blockDim.x - 1) / blockDim.x;
    accscalar_t prevHighProb = accZero;
    found = false;

    for (int chunk = 0; chunk < chunks && !found; ++chunk) {
      // All threads in bounds load a value
      int cat = chunk * blockDim.x + threadIdx.x;

      accscalar_t dist_val = cat < categories ?
                             static_cast<accscalar_t>(dist[curDist * stride_dist + cat * stride_categories]) / sum :
                             accZero;

      smem[threadIdx.x] = dist_val;
      __syncthreads();

      // Perform an inclusive prefix sum of the shared memory contents
      for (int offset = 1; offset < blockDim.x; offset *= 2) {
        accscalar_t val = accZero;

        if (threadIdx.x >= offset) {
          val = smem[threadIdx.x - offset] + smem[threadIdx.x];
        }

        __syncthreads();
        if (threadIdx.x >= offset) {
          smem[threadIdx.x] = val;
        }
        __syncthreads();
      }

      // Each thread will check to see if the sample falls in its bucket
      scalar_t curBucket =
          static_cast<scalar_t>(smem[threadIdx.x] + prevHighProb);
      scalar_t prevBucket = static_cast<scalar_t>(
          threadIdx.x == 0 ? prevHighProb
                          : smem[threadIdx.x - 1] + prevHighProb);
      bool inBucket =
          (cat < categories) &&
          (!(sample >= curBucket) &&
          (sample >= prevBucket) &&
          (dist_val > zero));

      if (inBucket) {
        atomicMax(&foundPos, cat);
        found = true;
      }

      // Store the previous scan's high value for future use
      prevHighProb = prevHighProb + smem[blockDim.x - 1];

      __syncthreads();
    }

    if (threadIdx.x == 0) {
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

  float *d_sample;
  hipMalloc((void**)&d_sample, sample_size_bytes);
  hipMemcpy(d_sample, sample, sample_size_bytes, hipMemcpyHostToDevice);

  float *d_distr;
  hipMalloc((void**)&d_distr, distr_size_bytes);
  hipMemcpy(d_distr, distr, distr_size_bytes, hipMemcpyHostToDevice);

  int *d_result;
  hipMalloc((void**)&d_result, result_size_bytes);

  int requiredThreads = GPU_NUM_THREADS;
  dim3 grid(512);
  dim3 block(requiredThreads);

  // warmup and verify
  sampleMultinomialOnce<float, float> <<<grid, block>>>(
      d_result, numDist, numCategories, d_sample, d_distr, numCategories, 1);

  sampleMultinomialOnce_cpu<float, float> (
      result_ref, numDist, numCategories, sample, distr, numCategories, 1);

  hipMemcpy(result, d_result, result_size_bytes, hipMemcpyDeviceToHost);

  int error = 0;
  for (int i = 0; i < numDist; i++) {
    if (abs(result[i] - result_ref[i]) > 1) {
      printf("results mismatch: %d %d %d\n", i, result[i], result_ref[i]);
      error = 1;
      break;
    }
  }
  printf("%s\n", error ? "FAIL" : "PASS");
  if (error) exit(1);

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    sampleMultinomialOnce<float, float> <<<grid, block>>>(
      d_result, numDist, numCategories, d_sample, d_distr, numCategories, 1);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of sampleMultinomialOnce kernel: %f (us)\n", (time * 1e-3f) / repeat);

  hipFree(d_result);
  hipFree(d_sample);
  hipFree(d_distr);

  free(result_ref);
  free(result);
  free(sample);
  free(distr);

  return 0;
}
