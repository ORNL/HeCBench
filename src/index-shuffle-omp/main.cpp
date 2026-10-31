/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * OpenMP target implementation of the MetaShuffling IndexShuffling benchmark.
 * It shares deterministic inputs and invariant-based verification with CUDA,
 * HIP, and SYCL.
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <omp.h>
#include <vector>

#include "../index-shuffle-cuda/reference.h"

namespace {

// Abort when a device allocation did not return a pointer.
void check_target(void *ptr, const char *name) {
  if (!ptr) {
    fprintf(stderr, "OpenMP target allocation failed for %s\n", name);
    std::exit(EXIT_FAILURE);
  }
}

// Copy host bytes to a device pointer, or abort on failure.
void copy_to_device(void *dest, const void *src, size_t bytes, int device) {
  if (omp_target_memcpy(dest, src, bytes, 0, 0, device,
                        omp_get_initial_device())) {
    fprintf(stderr, "OpenMP target host-to-device copy failed\n");
    std::exit(EXIT_FAILURE);
  }
}

// Copy device bytes to the host, or abort on failure.
void copy_to_host(void *dest, const void *src, size_t bytes, int device) {
  if (omp_target_memcpy(dest, src, bytes, 0, 0, omp_get_initial_device(),
                        device)) {
    fprintf(stderr, "OpenMP target device-to-host copy failed\n");
    std::exit(EXIT_FAILURE);
  }
}

// Zero the histogram and grid-barrier counters on the target. This matches the
// device-side reset timed by the other backends and avoids a host transfer per
// benchmark iteration.
void reset_counts(int *counts, int num_experts) {
#if defined(__NVCOMPILER)
#pragma omp target teams distribute parallel for is_device_ptr(counts)
#else
#pragma omp target teams distribute parallel for is_device_ptr(counts)          \
    firstprivate(num_experts)
#endif
  for (int i = 0; i < num_experts + 2; ++i)
    counts[i] = 0;
}

// Three target regions, in program order. A region finishes, and its device
// writes become visible, before the next region starts. That region boundary
// is the grid synchronization the CUDA kernel performs with a spin barrier.
//
//   1. One thread per token inserts expert scores into a sorted top-k list
//      and atomically counts the chosen experts.
//   2. One thread scans those counts into exclusive output offsets.
//   3. One thread per selected slot atomically claims the next offset of its
//      expert and stores the expert id with the original token id.
//
// nvc++ already captures the scalar loop bounds, so it rejects an added
// firstprivate clause. The other compilers need that clause to copy the
// bounds onto the device.
void index_shuffling(const float *scores, int num_tokens, int num_experts,
                     int top_k, int *counts, int *positions, int *chosen,
                     int *shuffled_experts, int *shuffled_tokens) {
#if defined(__NVCOMPILER)
#pragma omp target teams distribute parallel for is_device_ptr(scores, counts, chosen)
#else
#pragma omp target teams distribute parallel for is_device_ptr(                 \
    scores, counts, chosen) firstprivate(num_tokens, num_experts, top_k)
#endif
  // Higher scores shift the current top-k tail down. The list stays sorted,
  // so the following atomic updates count the final winners only.
  for (int token = 0; token < num_tokens; ++token) {
    float best_scores[4] = {-INFINITY, -INFINITY, -INFINITY, -INFINITY};
    int best_experts[4] = {-1, -1, -1, -1};
    for (int expert = 0; expert < num_experts; ++expert) {
      const float score = scores[token * num_experts + expert];
      for (int k = 0; k < top_k; ++k) {
        if (score > best_scores[k]) {
          for (int p = top_k - 1; p > k; --p) {
            best_scores[p] = best_scores[p - 1];
            best_experts[p] = best_experts[p - 1];
          }
          best_scores[k] = score;
          best_experts[k] = expert;
          break;
        }
      }
    }
    for (int k = 0; k < top_k; ++k) {
      const int expert = best_experts[k];
      chosen[token * top_k + k] = expert;
#pragma omp atomic update
      counts[expert]++;
    }
  }

#if defined(__NVCOMPILER)
#pragma omp target teams num_teams(1) thread_limit(1) is_device_ptr(counts, positions)
#else
#pragma omp target teams num_teams(1) thread_limit(1) is_device_ptr(           \
    counts, positions) firstprivate(num_experts)
#endif
  {
    // A single thread is enough: there are at most 128 experts.
    int prefix = 0;
    for (int expert = 0; expert < num_experts; ++expert) {
      positions[expert] = prefix;
      prefix += counts[expert];
    }
    counts[num_experts + 1] = prefix;
  }

#if defined(__NVCOMPILER)
#pragma omp target teams distribute parallel for is_device_ptr(positions, chosen, shuffled_experts, shuffled_tokens)
#else
#pragma omp target teams distribute parallel for is_device_ptr(                 \
    positions, chosen, shuffled_experts, shuffled_tokens)                       \
    firstprivate(num_tokens, top_k)
#endif
  // atomic capture takes the current offset and increments it in one step, so
  // two slots of the same expert cannot publish to the same output index.
  for (int flat = 0; flat < num_tokens * top_k; ++flat) {
    const int token = flat / top_k;
    const int expert = chosen[flat];
    int output;
#pragma omp atomic capture
    output = positions[expert]++;
    shuffled_experts[output] = expert;
    shuffled_tokens[output] = token;
  }
}

// Verify one shape, then time repeat target regions. Timing includes the histogram reset.
void run_case(int device, int num_tokens, int num_experts, int top_k,
              int repeat) {
  const int total = num_tokens * top_k;
  const size_t scores_bytes =
      static_cast<size_t>(num_tokens) * num_experts * sizeof(float);
  const size_t index_bytes = static_cast<size_t>(total) * sizeof(int);
  const size_t counts_bytes = static_cast<size_t>(num_experts + 2) * sizeof(int);
  std::vector<float> h_scores = make_routing_scores(num_tokens, num_experts);

  float *d_scores =
      static_cast<float *>(omp_target_alloc(scores_bytes, device));
  int *d_counts = static_cast<int *>(omp_target_alloc(counts_bytes, device));
  int *d_positions =
      static_cast<int *>(omp_target_alloc(num_experts * sizeof(int), device));
  int *d_chosen = static_cast<int *>(omp_target_alloc(index_bytes, device));
  int *d_shuffled_experts =
      static_cast<int *>(omp_target_alloc(index_bytes, device));
  int *d_shuffled_tokens =
      static_cast<int *>(omp_target_alloc(index_bytes, device));
  check_target(d_scores, "scores");
  check_target(d_counts, "counts");
  check_target(d_positions, "positions");
  check_target(d_chosen, "chosen");
  check_target(d_shuffled_experts, "shuffled experts");
  check_target(d_shuffled_tokens, "shuffled tokens");

  copy_to_device(d_scores, h_scores.data(), scores_bytes, device);
  reset_counts(d_counts, num_experts);
  index_shuffling(d_scores, num_tokens, num_experts, top_k, d_counts,
                  d_positions, d_chosen, d_shuffled_experts, d_shuffled_tokens);

  std::vector<int> h_counts(num_experts + 2);
  std::vector<int> h_shuffled_experts(total);
  std::vector<int> h_shuffled_tokens(total);
  copy_to_host(h_counts.data(), d_counts, counts_bytes, device);
  copy_to_host(h_shuffled_experts.data(), d_shuffled_experts, index_bytes,
               device);
  copy_to_host(h_shuffled_tokens.data(), d_shuffled_tokens, index_bytes,
               device);

  const int error =
      verify_index_shuffling(h_scores.data(), num_tokens, num_experts, top_k,
                             h_counts.data(), h_shuffled_experts.data(),
                             h_shuffled_tokens.data());
  printf("%s\n", error ? "FAIL" : "PASS");
  if (error)
    std::exit(EXIT_FAILURE);

  for (int i = 0; i < 10; ++i) {
    reset_counts(d_counts, num_experts);
    index_shuffling(d_scores, num_tokens, num_experts, top_k, d_counts,
                    d_positions, d_chosen, d_shuffled_experts,
                    d_shuffled_tokens);
  }

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) {
    reset_counts(d_counts, num_experts);
    index_shuffling(d_scores, num_tokens, num_experts, top_k, d_counts,
                    d_positions, d_chosen, d_shuffled_experts,
                    d_shuffled_tokens);
  }
  const auto end = std::chrono::steady_clock::now();
  const auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the index shuffle kernel "
         "(tokens %d, experts %d, topk %d): %f (us)\n",
         num_tokens, num_experts, top_k, (time * 1e-3f) / repeat);

  omp_target_free(d_scores, device);
  omp_target_free(d_counts, device);
  omp_target_free(d_positions, device);
  omp_target_free(d_chosen, device);
  omp_target_free(d_shuffled_experts, device);
  omp_target_free(d_shuffled_tokens, device);
}

} // namespace

// Run one shape on the default OpenMP target device: tokens, experts, top-k,
// repeat.
int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("Usage: %s <number of tokens> <number of experts> <top K> <repeat>\n",
           argv[0]);
    return 1;
  }
  const int num_tokens = std::atoi(argv[1]);
  const int num_experts = std::atoi(argv[2]);
  const int top_k = std::atoi(argv[3]);
  const int repeat = std::atoi(argv[4]);
  if (repeat <= 0 ||
      !valid_index_shuffling_shape(num_tokens, num_experts, top_k)) {
    printf("tokens and repeat must be positive; experts must be 16, 32, or 128; "
           "top K must be 1, 2, or 4; shape products must fit int\n");
    return 1;
  }
  if (omp_get_num_devices() < 1) {
    fprintf(stderr, "No OpenMP target device available\n");
    return 1;
  }
  run_case(omp_get_default_device(), num_tokens, num_experts, top_k, repeat);
  return 0;
}
