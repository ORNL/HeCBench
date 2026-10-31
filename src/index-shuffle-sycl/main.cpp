/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * SYCL implementation of the MetaShuffling IndexShuffling benchmark.
 * The input generator and invariant-based verifier are shared.
 */

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <vector>
#include <sycl/sycl.hpp>

#include "../index-shuffle-cuda/reference.h"

namespace {

// Two kernels on the in-order queue replace the CUDA kernel's spin barrier.
// Kernel completion makes every write visible to the next kernel, so the
// histogram and placement atomics only need relaxed atomicity.
//
//   1. One sub-group per token selects its top-k experts, then increments the
//      chosen experts' counts.
//   2. Each work-group scans the counts into exclusive output offsets; one
//      work-item per selected slot then claims the next offset of its expert
//      and writes the expert id and original token id there.
//
// counts and positions must be zero on entry. The atomic claim in step 2 makes
// the order inside an expert nondeterministic.
constexpr size_t kWorkGroupSize = 256;

// Return a lower bound on the tokens one work-group covers. Each token gets a
// segment of min(sub-group size, num_experts) lanes, and the sub-group size is
// chosen by the device compiler, so assume the largest supported size. That
// may launch extra segments, which stay idle, but never leaves a token
// uncovered.
size_t tokens_per_work_group(const sycl::queue &q, int num_experts) {
  const auto sizes =
      q.get_device().get_info<sycl::info::device::sub_group_sizes>();
  const size_t max_sub_group_size = *std::max_element(sizes.begin(), sizes.end());
  return kWorkGroupSize /
         std::min(max_sub_group_size, static_cast<size_t>(num_experts));
}

void index_shuffling(sycl::queue &q, size_t tokens_per_group,
                     const float *scores, int num_tokens, int num_experts,
                     int top_k, int *counts, int *positions, int *chosen,
                     int *shuffled_experts, int *shuffled_tokens) {
  const int total = num_tokens * top_k;
  const size_t num_groups = (num_tokens + tokens_per_group - 1) / tokens_per_group;

  // A sub-group is split into segments of width min(lanes, num_experts); both
  // are powers of two, so segments tile the sub-group and 16 experts fill all
  // lanes of a subgroup with four tokens. Each segment lane reads
  // experts seg_lane, seg_lane + width, ... so a segment loads one token's
  // scores with adjacent addresses. Every round, each lane keeps its best
  // expert not yet chosen, and an XOR butterfly over the segment picks the
  // highest score, breaking ties toward the smaller expert index like the
  // verifier. XOR offsets below the width never leave the segment.
  q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<1>(num_groups * kWorkGroupSize, kWorkGroupSize),
        [=](sycl::nd_item<1> item) {
          const sycl::sub_group sg = item.get_sub_group();
          const int lane = sg.get_local_linear_id();
          const int lanes = sg.get_local_linear_range();
          const int width = sycl::min(lanes, num_experts);
          const int segments = lanes / width;
          const int seg_lane = lane & (width - 1);
          const int token =
              (item.get_group_linear_id() * sg.get_group_linear_range() +
               sg.get_group_linear_id()) * segments + lane / width;
          // Segments past the last token still join the shuffles, which need
          // the whole sub-group, but read and write nothing.
          const bool active = token < num_tokens;

          const float *row = scores + token * num_experts;
          int picked[4] = {-1, -1, -1, -1};
          for (int k = 0; k < top_k; ++k) {
            float best_score = -INFINITY;
            int best_expert = INT_MAX;
            for (int expert = seg_lane; active && expert < num_experts;
                 expert += width) {
              bool taken = false;
              for (int p = 0; p < k; ++p)
                taken |= picked[p] == expert;
              const float score = row[expert];
              if (!taken && (score > best_score ||
                             (score == best_score && expert < best_expert))) {
                best_score = score;
                best_expert = expert;
              }
            }
            for (int offset = width / 2; offset > 0; offset /= 2) {
              const float other_score =
                  sycl::permute_group_by_xor(sg, best_score, offset);
              const int other_expert =
                  sycl::permute_group_by_xor(sg, best_expert, offset);
              if (other_score > best_score ||
                  (other_score == best_score && other_expert < best_expert)) {
                best_score = other_score;
                best_expert = other_expert;
              }
            }
            picked[k] = best_expert;
          }

          // Segment lane k saves and counts the k-th winner. The next kernel
          // reads these counts only after this kernel has completed.
          for (int k = 0; k < top_k; ++k) {
            if (active && seg_lane == k) {
              const int expert = picked[k];
              chosen[token * top_k + k] = expert;
              sycl::atomic_ref<int, sycl::memory_order::relaxed,
                               sycl::memory_scope::device,
                               sycl::access::address_space::global_space>
                  count(counts[expert]);
              count.fetch_add(1);
            }
          }
        });
  });

  // Every work-group scans the at most 128 counts into local memory, so
  // offsets[e] is the first output index of expert e. Redoing this small scan
  // per group is cheaper than a separate single-work-item launch.
  //
  // fetch_add on positions[e] (zeroed with counts) returns this slot's index
  // within the expert and advances the cursor for the next slot. The writes
  // therefore form one contiguous run per expert without a stable order inside
  // the run.
  const size_t place_groups = (total + kWorkGroupSize - 1) / kWorkGroupSize;
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<int, 1> offsets(sycl::range<1>(num_experts), h);
    h.parallel_for(
        sycl::nd_range<1>(place_groups * kWorkGroupSize, kWorkGroupSize),
        [=](sycl::nd_item<1> item) {
          int *local_offsets = &offsets[0];
          sycl::joint_exclusive_scan(item.get_group(), counts,
                                     counts + num_experts, local_offsets,
                                     sycl::plus<int>());
          const int flat = item.get_global_linear_id();
          // The running total is also the number of selected tokens.
          if (flat == 0)
            counts[num_experts + 1] =
                local_offsets[num_experts - 1] + counts[num_experts - 1];
          if (flat >= total)
            return;
          const int token = flat / top_k;
          const int expert = chosen[flat];
          sycl::atomic_ref<int, sycl::memory_order::relaxed,
                           sycl::memory_scope::device,
                           sycl::access::address_space::global_space>
              position(positions[expert]);
          const int output = local_offsets[expert] + position.fetch_add(1);
          shuffled_experts[output] = expert;
          shuffled_tokens[output] = token;
        });
  });
}

// Verify one shape, then time repeat submissions. Timing includes the histogram reset.
void run_case(sycl::queue &q, int num_tokens, int num_experts, int top_k, int repeat) {
  const int total = num_tokens * top_k;
  const size_t scores_bytes = (size_t)num_tokens * num_experts * sizeof(float);
  const size_t index_bytes = (size_t)total * sizeof(int);
  const size_t counts_bytes = (size_t)(num_experts + 2) * sizeof(int);

  // positions follows counts in one allocation so one memset resets both.
  const size_t reset_bytes = counts_bytes + num_experts * sizeof(int);
  const size_t tokens_per_group = tokens_per_work_group(q, num_experts);
  std::vector<float> h_scores = make_routing_scores(num_tokens, num_experts);

  float *d_scores = sycl::malloc_device<float>(num_tokens * num_experts, q);
  int *d_counts = sycl::malloc_device<int>(2 * num_experts + 2, q);
  int *d_positions = d_counts + num_experts + 2;
  int *d_chosen = sycl::malloc_device<int>(total, q);
  int *d_shuffled_experts = sycl::malloc_device<int>(total, q);
  int *d_shuffled_tokens = sycl::malloc_device<int>(total, q);
  if (!d_scores || !d_counts || !d_chosen ||
      !d_shuffled_experts || !d_shuffled_tokens)
    throw std::bad_alloc();

  q.memcpy(d_scores, h_scores.data(), scores_bytes);
  q.memset(d_counts, 0, reset_bytes);
  index_shuffling(q, tokens_per_group, d_scores, num_tokens, num_experts,
                  top_k, d_counts, d_positions, d_chosen,
                  d_shuffled_experts, d_shuffled_tokens);

  std::vector<int> h_counts(num_experts + 2);
  std::vector<int> h_shuffled_experts(total);
  std::vector<int> h_shuffled_tokens(total);
  q.memcpy(h_counts.data(), d_counts, counts_bytes);
  q.memcpy(h_shuffled_experts.data(), d_shuffled_experts, index_bytes);
  q.memcpy(h_shuffled_tokens.data(), d_shuffled_tokens, index_bytes);
  q.wait();

  const int error =
      verify_index_shuffling(h_scores.data(), num_tokens, num_experts, top_k,
                             h_counts.data(), h_shuffled_experts.data(),
                             h_shuffled_tokens.data());
  printf("%s\n", error ? "FAIL" : "PASS");
  if (error)
    std::exit(EXIT_FAILURE);

  for (int i = 0; i < 100; ++i) {
    q.memset(d_counts, 0, reset_bytes);
    index_shuffling(q, tokens_per_group, d_scores, num_tokens, num_experts,
                    top_k, d_counts, d_positions, d_chosen,
                    d_shuffled_experts, d_shuffled_tokens);
  }
  q.wait();

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) {
    q.memset(d_counts, 0, reset_bytes);
    index_shuffling(q, tokens_per_group, d_scores, num_tokens, num_experts,
                    top_k, d_counts, d_positions, d_chosen,
                    d_shuffled_experts, d_shuffled_tokens);
  }
  q.wait();
  const auto end = std::chrono::steady_clock::now();
  const auto time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of the index shuffle kernel "
         "(tokens %d, experts %d, topk %d): %f (us)\n",
         num_tokens, num_experts, top_k, (time * 1e-3f) / repeat);

  sycl::free(d_scores, q);
  sycl::free(d_counts, q);
  sycl::free(d_chosen, q);
  sycl::free(d_shuffled_experts, q);
  sycl::free(d_shuffled_tokens, q);
}

} // namespace

// Run one shape on the selected device: tokens, experts, top-k, and repeat
// count.
int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("Usage: %s <number of tokens> <number of experts> <top K> <repeat>\n",
           argv[0]);
    printf("  experts must be 16, 32, or 128; top K must be 1, 2, or 4\n");
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

  try {
#if defined(USE_GPU)
    sycl::queue q{sycl::gpu_selector_v, sycl::property::queue::in_order{}};
#else
    sycl::queue q{sycl::cpu_selector_v,
                  sycl::property::queue::in_order{}};
#endif
    printf("Running on %s\n", q.get_device().get_info<sycl::info::device::name>()
                                  .c_str());
    run_case(q, num_tokens, num_experts, top_k, repeat);
  } catch (const sycl::exception &e) {
    fprintf(stderr, "SYCL error: %s\n", e.what());
    return 1;
  } catch (const std::exception &e) {
    fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }
  return 0;
}
