/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * IndexShuffling reference. Placement order within an
 * expert is nondeterministic (atomic ranks), so checks are invariant-based.
 */

#pragma once

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// Return whether every supported shape product fits the int indexing used by
// the benchmark's host and device buffers.
inline bool valid_index_shuffling_shape(int num_tokens, int num_experts,
                                        int top_k) {
  if (num_tokens <= 0 ||
      !(num_experts == 16 || num_experts == 32 || num_experts == 128) ||
      !(top_k == 1 || top_k == 2 || top_k == 4))
    return false;
  return num_tokens <= INT_MAX / num_experts &&
         num_tokens <= INT_MAX / top_k;
}

// Build the same tie-free score matrix on every backend. Seed 19937.
inline std::vector<float> make_routing_scores(int num_tokens, int num_experts) {
  std::vector<float> scores(num_tokens * num_experts);
  std::mt19937 gen(19937);
  for (int t = 0; t < num_tokens; ++t) {
    std::vector<float> row(num_experts);
    for (int e = 0; e < num_experts; ++e)
      row[e] = static_cast<float>(e);
    std::shuffle(row.begin(), row.end(), gen);
    for (int e = 0; e < num_experts; ++e)
      scores[t * num_experts + e] = row[e] + 0.01f * static_cast<float>(t);
  }
  return scores;
}

// Store each token's top-k expert ids, breaking ties toward the smaller index.
inline void select_topk_experts(const float *scores, int num_tokens,
                                int num_experts, int top_k,
                                std::vector<int> &chosen_experts) {
  chosen_experts.assign(num_tokens * top_k, -1);
  for (int t = 0; t < num_tokens; ++t) {
    const float *row = scores + t * num_experts;
    for (int k = 0; k < top_k; ++k) {
      int best_e = -1;
      float best_s = -INFINITY;
      for (int e = 0; e < num_experts; ++e) {
        bool already = false;
        for (int p = 0; p < k; ++p) {
          if (chosen_experts[t * top_k + p] == e) {
            already = true;
            break;
          }
        }
        if (already)
          continue;
        if (row[e] > best_s || (row[e] == best_s && (best_e < 0 || e < best_e))) {
          best_s = row[e];
          best_e = e;
        }
      }
      chosen_experts[t * top_k + k] = best_e;
    }
  }
}

// Return 0 when counts, contiguous expert runs, and the token permutation match.
// Order within an expert is unchecked because atomic ranks are nondeterministic.
inline int verify_index_shuffling(const float *scores, int num_tokens,
                                  int num_experts, int top_k,
                                  const int *token_count_per_expert,
                                  const int *shuffled_expert_indices,
                                  const int *shuffled_token_indices) {
  std::vector<int> chosen;
  select_topk_experts(scores, num_tokens, num_experts, top_k, chosen);

  std::vector<int> expected_counts(num_experts, 0);
  for (int i = 0; i < num_tokens * top_k; ++i) {
    expected_counts[chosen[i]]++;
  }

  for (int e = 0; e < num_experts; ++e) {
    if (token_count_per_expert[e] != expected_counts[e]) {
      printf("FAIL: token_count[%d]=%d expected %d\n", e,
             token_count_per_expert[e], expected_counts[e]);
      return 1;
    }
  }

  const int total = num_tokens * top_k;
  if (token_count_per_expert[num_experts + 1] != total) {
    printf("FAIL: num_selected_tokens=%d expected %d\n",
           token_count_per_expert[num_experts + 1], total);
    return 1;
  }

  // Expert runs must be contiguous and match the counts.
  int offset = 0;
  for (int e = 0; e < num_experts; ++e) {
    for (int i = 0; i < expected_counts[e]; ++i) {
      if (shuffled_expert_indices[offset + i] != e) {
        printf("FAIL: shuffled_expert_indices[%d]=%d expected %d\n",
               offset + i, shuffled_expert_indices[offset + i], e);
        return 1;
      }
    }
    offset += expected_counts[e];
  }

  // Every (token, top-k slot) must appear exactly once, and the expert at the
  // shuffled position must be one of that token's chosen experts.
  std::vector<int> seen(num_tokens * top_k, 0);
  for (int i = 0; i < total; ++i) {
    const int token = shuffled_token_indices[i];
    const int expert = shuffled_expert_indices[i];
    if (token < 0 || token >= num_tokens) {
      printf("FAIL: shuffled_token_indices[%d]=%d out of range\n", i, token);
      return 1;
    }
    int slot = -1;
    for (int k = 0; k < top_k; ++k) {
      if (chosen[token * top_k + k] == expert) {
        slot = k;
        break;
      }
    }
    if (slot < 0) {
      printf("FAIL: token %d at pos %d has expert %d not in its top-%d\n",
             token, i, expert, top_k);
      return 1;
    }
    const int key = token * top_k + slot;
    if (seen[key]) {
      printf("FAIL: duplicate placement for token %d expert-slot %d\n", token,
             slot);
      return 1;
    }
    seen[key] = 1;
  }
  for (int i = 0; i < total; ++i) {
    if (!seen[i]) {
      printf("FAIL: missing placement for flattened index %d\n", i);
      return 1;
    }
  }
  return 0;
}
