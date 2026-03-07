#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cuda.h>

#include "reference.h"

template <class T>
__device__ T clamp(T value, T lower, T upper) { return min(max(value, lower), upper); }

template <class T, int BINS, int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE)
__global__
void findTopK(int*__restrict__ indices_, 
              int*__restrict__ count_, 
              const T*__restrict__ scores_,
              const float threshold,
              const int classwise_topK,
              const size_t num_classes,
              const size_t num_priors)
{
  /* We need to sort boxes based on their confidence scores. The confidence scores fall in
   * the range [0.0, 1.0]. We break the range into bins and perform count sort. This is an
   * approximate algorithm.
   *
   * Each block handles a particular class of a particular batch item.
   */
  const auto c = blockIdx.x;
  const auto b = blockIdx.y;

  // indices: [batch_size, num_classes, classwise_topK]
  // count: [batch_size, num_classes]
  // scores: [batch_size, num_classes, num_priors]

  auto indices = indices_ + (b * num_classes + c) * classwise_topK;
  auto count = count_ + b * num_classes + c;
  auto scores = scores_ + (b * num_classes + c) * num_priors;

  /* We do not require a large number of bins to find the top K confidence scores. We will use
   * a reasonable number of bins which will fit in the shared memory.
   *
   * Note that smaller scores will have a smaller index, i.e. the `bins` are ordered in
   * ascending order.
   */

  __shared__ int bins[BINS];

  #pragma unroll
  for (int unroll = 0; unroll < BINS / BLOCK_SIZE; unroll++)
    bins[unroll * BLOCK_SIZE + threadIdx.x] = 0;

  __syncthreads();

  for (int i = threadIdx.x; i < num_priors; i = i + BLOCK_SIZE)
  {
    const float confidence = scores[i];
    if (confidence > threshold)
    {
      float conf_scaled = __fdividef(confidence - threshold, 1.f - threshold);
      int bin_index = conf_scaled * BINS;

      /* We store counts of confidence scores in the bins. Our ultimate goal is to store the indices
       * of the `classwise_topK` confidence values in the `indices` array.
       *
       * We use a little trick to parallelize the process of filling up the `indices` array.
       * We want every thread in the block to participate in the process. To do so, we want the
       * bins array to be shifted by one place to the left. We will be computing the suffix sum
       * of the bins array later. Details and reasons for doing so will be explained later.
       */
      bin_index = clamp<int>(bin_index, 0, BINS - 1) - 1; // shift left by one

      if (bin_index >= 0)
        atomicAdd(&bins[bin_index], 1);
    }
  }

  __syncthreads();

  constexpr int WARP_SIZE = 32; /* must be equal to warpSize */

  if (threadIdx.x < WARP_SIZE)
  {
    /* We can compute suffix sum of an array in groups of N numbers.
     * Let N be 4 for this example.
     *
     * 1) Last 4 numbers
     *                      1   2   3   4   |   5   6   7   8   |   9   10  11  12
     * group suffix sum:                                            42  33  23  12
     *
     * 2) Middle 4 numbers
     *                      1   2   3   4   |   5   6   7   8   |   9   10  11  12
     * group suffix sum:                    |   26  21  15  8   |
     *
     * We add `42` (first element in the previous group) to each element to get:
     *
     *                      1   2   3   4   |   5   6   7   8   |   9   10  11  12
     *                                      |   68  63  57  50  |   42  33  23  12
     * 3) First 4 numbers
     *
     *                      1   2   3   4   |   5   6   7   8   |   9   10  11  12
     * group suffix sum:    10  9   7   4   |
     *
     * We add `68` (first element in the previous group) to each element to get:
     *
     *                      1   2   3   4   |   5   6   7   8   |   9   10  11  12
     * group suffix sum:    78  77  75  72  |   68  63  57  50  |   42  33  23  12
     *
     * What we are left with now is the suffix sum of the entire array.
     *
     * We use the aforementioned logic in the code below but work in groups of `warpSize`.
     */

    /* We calculate suffix sums WARP_SIZE elements at a time starting from the right end.
     * Hence, we will need BINS / WARP_SIZE number of iterations.
     *
     * Each iteration uses shuffle instructions to exchange data between threads. Shuffle
     * instructions cannot be used in warp-divergent code. If the bins are a multiple of
     * the warpSize, all the threads in the warp will participate.
     */
    static_assert(BINS % WARP_SIZE == 0, "number of bins must be a multiple of warp size");

    const int thread_id = threadIdx.x;
    const int inverse_lane_id = WARP_SIZE - thread_id - 1;

    int previous_group_first_element = 0;
    for (int iter = BINS / WARP_SIZE - 1; iter >= 0; iter--)
    {
      const int idx = iter * WARP_SIZE + thread_id;
      auto value = bins[idx];

      for (int i = 1; i < WARP_SIZE; i *= 2)
      {
        auto n = __shfl_down_sync(0xFFFFFFFF, value, i);
        if (inverse_lane_id >= i)
          value += n;
      }

      value += previous_group_first_element;
      bins[idx] = value;

      previous_group_first_element = __shfl_sync(0xFFFFFFFF, value, 0);
    }
  }

  if (threadIdx.x == 0) *count = 0;

  __syncthreads();

  for (int i = threadIdx.x; i < num_priors; i = i + BLOCK_SIZE)
  {
    const float confidence = scores[i];
    if (confidence > threshold)
    {
      float conf_scaled = __fdividef(confidence - threshold, 1.f - threshold);
      int bin_index = conf_scaled * BINS;
      bin_index = clamp<int>(bin_index, 0, BINS - 1);

      /* This bounding box is eligible to be selected unless it does not fall in
       * the `classwise_topK`. If it did, we would have to compute the location where it needs
       * to be stored.
       *
       * Suppose we had just 4 bins and say the following were the counts:
       * BIN0 2
       * BIN1 1
       * BIN2 3
       * BIN3 0 (last bin is always zero as we shift left by one while populating the bins)
       *
       * We will try our best to store the boxes in a sorted order in the `indices` array.
       * This requires that the boxes in later bins (higher confidence scores) must be
       * stored earlier.
       *
       * We compute the suffix sum of the array. This gives us:
       * BIN0 6
       * BIN1 4
       * BIN2 3
       * BIN3 0
       *
       * The bins now give us the location in the `indices` array from which the indices of the
       * scores corresponding to that bin would be stored. We atomically increment the bin count
       * everytime we store a box corresponding to that bin. Therefore, the value in the bins
       * gives the index in the `indices` array where the next box corresponding to that bin  must
       * be put.
       */

      const int idx = atomicAdd(&bins[bin_index], 1);
      if (idx < classwise_topK)
      {
        indices[idx] = i;
        atomicAdd(&count[0], 1);
      }
    }
  }
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const float threshold = 0.4f;
  const int classwise_topK = 10;
  const size_t num_classes = 1000;
  const size_t num_priors = 4096;
  const int batch_size = 128;
  const int block_size = 256; 
  
  dim3 grids (num_classes, batch_size);
  dim3 blocks (block_size, 1);

  size_t indices_size = batch_size * num_classes * classwise_topK;
  size_t indices_size_bytes = indices_size * sizeof(int);

  size_t count_size = batch_size * num_classes;
  size_t count_size_bytes = count_size * sizeof(int);

  size_t scores_size = batch_size * num_classes * num_priors;
  size_t scores_size_bytes = scores_size * sizeof(int);

  float *scores = (float*) malloc (scores_size_bytes);
  int *count = (int*) malloc (count_size_bytes);
  int *count_ref = (int*) malloc (count_size_bytes);
  memset(count_ref, 0, count_size_bytes);
  int *indices = (int*) malloc (indices_size_bytes);
  int *indices_ref = (int*) malloc (indices_size_bytes);

  float *d_scores;
  int *d_count;
  int *d_indices; 
  cudaMalloc((void**)&d_indices, indices_size_bytes);
  cudaMemset(d_indices, 0, indices_size_bytes);
  cudaMalloc((void**)&d_count, count_size_bytes);
  cudaMalloc((void**)&d_scores, scores_size_bytes);

  srand(123);
  for (int b = 0; b < batch_size; b++) {
    for (size_t c = 0; c < num_classes; c++) {
      float *s = scores + b * num_classes * num_priors + c * num_priors;
      for (size_t p = 0; p < num_priors; p++)
        s[p] = p * 1.0 / num_priors;
      for (int i = num_priors-1; i > 0; i--)
        std::swap(s[i], s[rand() % (i+1)]);
    }
  }
  cudaMemcpy(d_scores, scores, scores_size_bytes, cudaMemcpyHostToDevice);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    findTopK<float, 2048, block_size> <<<grids, blocks>>> (
      d_indices, d_count, d_scores, threshold, classwise_topK, num_classes, num_priors);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (ms)\n", time * 1e-6f / repeat);

  cudaMemcpy(indices, d_indices, indices_size_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(count, d_count, count_size_bytes, cudaMemcpyDeviceToHost);

  reference<float, 2048>(indices_ref, count_ref, scores, threshold, classwise_topK, batch_size, num_classes, num_priors);

  unsigned checksum = 0; 
  for (int b = 0; b < batch_size; b++) {
    for (size_t c = 0; c < num_classes; c++) {
      size_t offset = b * num_classes * classwise_topK + c * classwise_topK;
      auto topK = indices + offset; 
      auto topK_ref = indices_ref + offset;
      std::sort(topK, topK+classwise_topK);
      std::sort(topK_ref, topK_ref+classwise_topK);
      checksum += memcmp(topK, topK_ref, sizeof(int)*classwise_topK);
    }
  }
  checksum += memcmp(count, count_ref, count_size_bytes);
  printf("%s\n", checksum == 0 ? "PASS" : "FAIL");
  if (checksum != 0) exit(1);

  cudaFree(d_indices);
  cudaFree(d_count);
  cudaFree(d_scores);
  free(indices);
  free(indices_ref);
  free(count);
  free(count_ref);
  free(scores);

  return 0;
}
