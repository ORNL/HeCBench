/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * Adapted from Meta Superintelligence Labs Kernels (MSLK) index_shuffling:
 * https://github.com/meta-pytorch/MSLK/blob/main/csrc/moe/index_shuffling.cu
 *
 * Benchmark for MetaShuffling IndexShuffling: fused top-k, per-expert
 * histogram, and expert-order token-index placement used by Llama 4 MoE.
 */

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "kernels.h"
#include "reference.h"

// Verify one shape, then time repeat launches. Timing includes the histogram reset.
// split selects the two-kernel path; otherwise the cooperative kernel runs.
void run_case(int num_tokens, int num_experts, int top_k, int repeat,
              bool split) {
  const int total = num_tokens * top_k;
  const size_t scores_bytes = (size_t)num_tokens * num_experts * sizeof(float);
  const size_t index_bytes = (size_t)total * sizeof(int);
  const size_t counts_bytes = (size_t)(num_experts + 2) * sizeof(int);
  const size_t positions_bytes = (size_t)num_experts * sizeof(int);
  // positions follows counts. The split path's placement atomics start at
  // zero, so one memset clears both arrays. The fused path does not use
  // positions.
  const size_t reset_bytes = counts_bytes + (split ? positions_bytes : 0);

  std::vector<float> h_scores = make_routing_scores(num_tokens, num_experts);

  float *d_scores = nullptr;
  int *d_counts = nullptr;
  int *d_shuffled_experts = nullptr;
  int *d_shuffled_tokens = nullptr;
  int *d_buf_experts = nullptr;
  int *d_buf_tokens = nullptr;

  CHECK(cudaMalloc(&d_scores, scores_bytes));
  CHECK(cudaMalloc(&d_counts, counts_bytes + positions_bytes));
  CHECK(cudaMalloc(&d_shuffled_experts, index_bytes));
  CHECK(cudaMalloc(&d_shuffled_tokens, index_bytes));
  CHECK(cudaMalloc(&d_buf_experts, index_bytes));
  CHECK(cudaMalloc(&d_buf_tokens, index_bytes));
  int *d_positions = d_counts + num_experts + 2;
  CHECK(cudaMemcpy(d_scores, h_scores.data(), scores_bytes,
                   cudaMemcpyHostToDevice));

  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, 0));
  const int num_sms = prop.multiProcessorCount;
  cudaStream_t stream = 0;

  auto launch = [&]() {
    if (split) {
      dispatch_index_shuffling_split(
          d_scores, num_tokens, num_experts, top_k, d_counts, d_positions,
          d_buf_experts, d_shuffled_experts, d_shuffled_tokens, prop.warpSize,
          stream);
    } else {
      dispatch_index_shuffling(d_scores, num_tokens, num_experts, top_k,
                               d_counts, d_shuffled_experts, d_shuffled_tokens,
                               d_buf_experts, d_buf_tokens, num_sms,
                               prop.warpSize, stream);
    }
  };

  CHECK(cudaMemset(d_counts, 0, reset_bytes));
  launch();

  std::vector<int> h_counts(num_experts + 2);
  std::vector<int> h_shuffled_experts(total);
  std::vector<int> h_shuffled_tokens(total);
  CHECK(cudaMemcpy(h_counts.data(), d_counts, counts_bytes,
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h_shuffled_experts.data(), d_shuffled_experts, index_bytes,
                   cudaMemcpyDeviceToHost));
  CHECK(cudaMemcpy(h_shuffled_tokens.data(), d_shuffled_tokens, index_bytes,
                   cudaMemcpyDeviceToHost));

  const int error =
      verify_index_shuffling(h_scores.data(), num_tokens, num_experts, top_k,
                             h_counts.data(), h_shuffled_experts.data(),
                             h_shuffled_tokens.data());
  printf("%s\n", error ? "FAIL" : "PASS");
  if (error) {
    CHECK(cudaFree(d_scores));
    CHECK(cudaFree(d_counts));
    CHECK(cudaFree(d_shuffled_experts));
    CHECK(cudaFree(d_shuffled_tokens));
    CHECK(cudaFree(d_buf_experts));
    CHECK(cudaFree(d_buf_tokens));
    exit(EXIT_FAILURE);
  }

  // Warmup
  for (int i = 0; i < 100; ++i) {
    CHECK(cudaMemsetAsync(d_counts, 0, reset_bytes, stream));
    launch();
  }
  CHECK(cudaDeviceSynchronize());

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) {
    CHECK(cudaMemsetAsync(d_counts, 0, reset_bytes, stream));
    launch();
  }
  CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the index shuffle kernel "
         "(tokens %d, experts %d, topk %d, %s): %f (us)\n",
         num_tokens, num_experts, top_k, split ? "split" : "fused",
         (time * 1e-3f) / repeat);

  CHECK(cudaFree(d_scores));
  CHECK(cudaFree(d_counts));
  CHECK(cudaFree(d_shuffled_experts));
  CHECK(cudaFree(d_shuffled_tokens));
  CHECK(cudaFree(d_buf_experts));
  CHECK(cudaFree(d_buf_tokens));
}

// Run one shape given as tokens, experts, top-k, and repeat count.
// An optional last argument selects fused (default) or split.
int main(int argc, char *argv[]) {
  if (argc != 5 && argc != 6) {
    printf("Usage: %s <number of tokens> <number of experts> <top K> <repeat> "
           "[fused|split]\n",
           argv[0]);
    printf("  experts must be 16, 32, or 128; top K must be 1, 2, or 4\n");
    printf("  fused (default) is one cooperative kernel; split is histogram "
           "then placement\n");
    return 1;
  }
  const int num_tokens = atoi(argv[1]);
  const int num_experts = atoi(argv[2]);
  const int top_k = atoi(argv[3]);
  const int repeat = atoi(argv[4]);
  bool split = false;
  if (argc == 6) {
    if (std::strcmp(argv[5], "split") == 0)
      split = true;
    else if (std::strcmp(argv[5], "fused") != 0) {
      printf("implementation must be fused or split\n");
      return 1;
    }
  }
  if (repeat <= 0 ||
      !valid_index_shuffling_shape(num_tokens, num_experts, top_k)) {
    printf("tokens and repeat must be positive; experts must be 16, 32, or 128; "
           "top K must be 1, 2, or 4; shape products must fit int\n");
    return 1;
  }
  run_case(num_tokens, num_experts, top_k, repeat, split);
  return 0;
}
