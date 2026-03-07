#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <cuda.h>
#include "kernels.h"
#include "reference.h"

void topk_softmax(int num_tokens, int num_experts, int topk, int repeat)
{
  const int index_size = num_tokens * topk;
  const int index_size_bytes = index_size * sizeof(int);

  const int weight_size = num_tokens * topk;
  const int weight_size_bytes = weight_size * sizeof(float);

  const int output_size = num_tokens * num_experts;
  const int output_size_bytes = output_size * sizeof(float);

  float *topk_weights = (float*) malloc (weight_size_bytes);
  int *topk_indices = (int*) malloc (index_size_bytes);
  int *token_expert_indices = (int*) malloc (index_size_bytes);
  float *gating_output = (float*) malloc (output_size_bytes);

  float *softmax_workspace = (float*) malloc (output_size_bytes);
  float *topk_weights_ref = (float*) malloc (weight_size_bytes);
  int *topk_indices_ref = (int*) malloc (index_size_bytes);
  int *token_expert_indices_ref = (int*) malloc (index_size_bytes);

  std::mt19937 gen(19937);
  std::uniform_int_distribution<> distrib(-1e5, 1e5);
  for (int i = 0; i < output_size; i++) {
    gating_output[i] = distrib(gen);
  }
  for (int i = 0; i < topk; i++) {
    for (int j = 0; j < num_tokens; j++) {
      topk_indices_ref[i * num_tokens + j] =
      topk_indices[i * num_tokens + j] = abs(distrib(gen)) % num_experts;
    }
  }

  moeSoftmax_reference(
          gating_output,
          nullptr,
          softmax_workspace,
          num_tokens,
          num_experts);

  moeTopK_reference(
          softmax_workspace,
          nullptr,
          topk_weights_ref,
          topk_indices_ref,
          token_expert_indices_ref,
          num_tokens,
          num_experts,
          topk,
          0,  // start_expert
          num_experts);

  float *d_topk_weights;
  cudaMalloc(&d_topk_weights, weight_size_bytes);

  int *d_topk_indices;
  cudaMalloc(&d_topk_indices, index_size_bytes);

  int *d_token_expert_indices;
  cudaMalloc(&d_token_expert_indices, index_size_bytes);

  float *d_gating_output;
  cudaMalloc(&d_gating_output, output_size_bytes);

  cudaMemcpy(d_gating_output, gating_output, output_size_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(d_topk_indices, topk_indices, index_size_bytes, cudaMemcpyHostToDevice);

  float *d_softmax_workspace;
  cudaMalloc(&d_softmax_workspace, output_size_bytes);

  static constexpr int TPB = 256;

  moeSoftmax<TPB><<<num_tokens, TPB>>>(
          d_gating_output,
          nullptr,
          d_softmax_workspace,
          num_experts);

  moeTopK<TPB><<<num_tokens, TPB>>>(
          d_softmax_workspace,
          nullptr,
          d_topk_weights,
          d_topk_indices,
          d_token_expert_indices,
          num_experts,
          topk,
          0,  // start_expert
          num_experts);

  cudaMemcpy(topk_weights, d_topk_weights, weight_size_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(topk_indices, d_topk_indices, index_size_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(token_expert_indices, d_token_expert_indices, index_size_bytes, cudaMemcpyDeviceToHost);

  int error = memcmp(topk_indices, topk_indices_ref, index_size_bytes);
  error += memcmp(token_expert_indices, token_expert_indices_ref, index_size_bytes);
  for (int i = 0; i < weight_size; i++) {
    if (fabsf(topk_weights[i] - topk_weights_ref[i]) > 1e-3f)  {
      error = 1;
      break;
    }
  }
  printf("%s\n", error ? "FAIL" : "PASS");

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    moeSoftmax<TPB><<<num_tokens, TPB>>>(
            d_gating_output,
            nullptr,
            d_softmax_workspace,
            num_experts);

    moeTopK<TPB><<<num_tokens, TPB>>>(
            d_softmax_workspace,
            nullptr,
            d_topk_weights,
            d_topk_indices,
            d_token_expert_indices,
            num_experts,
            topk,
            0,  // start_expert
            num_experts);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of kernels: %f (us)\n", (time * 1e-3f) / repeat);

  cudaFree(d_topk_weights);
  cudaFree(d_topk_indices);
  cudaFree(d_token_expert_indices);
  cudaFree(d_gating_output);
  cudaFree(d_softmax_workspace);

  free(topk_weights);
  free(topk_indices);
  free(token_expert_indices);
  free(gating_output);

  free(topk_weights_ref);
  free(topk_indices_ref);
  free(token_expert_indices_ref);
  free(softmax_workspace);
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: %s <number of tokens> <number of experts> <top K> <repeat>\n", argv[0]);
    return 1;
  }
  const int num_tokens = atoi(argv[1]);
  const int num_experts = atoi(argv[2]);
  const int topk = atoi(argv[3]);
  const int repeat = atoi(argv[4]);
  topk_softmax(num_tokens, num_experts, topk, repeat);
  if (error) return 1;
  return 0;
}
