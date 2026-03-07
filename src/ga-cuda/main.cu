#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <cuda.h>
#include "reference.h"

__global__ 
void ga(const char *__restrict__ target,
        const char *__restrict__ query,
              char *__restrict__ batch_result,
              uint32_t length,
              int query_sequence_length,
              int coarse_match_length,
              int coarse_match_threshold,
              int current_position)
{
  uint tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid > length) return;
  bool match = false;
  int max_length = query_sequence_length - coarse_match_length;

  for (int i = 0; i <= max_length; i++) {
    int distance = 0;
    for (int j = 0; j < coarse_match_length; j++) {
      if (target[current_position + tid + j] != query[i + j]) {
        distance++;
      }
    }

    if (distance < coarse_match_threshold) {
      match = true;
      break;
    }
  }
  if (match) {
    batch_result[tid] = 1;
  }
}

int main(int argc, char* argv[]) 
{
  if (argc != 5) {
    printf("Usage: %s <target sequence length> <query sequence length> "
           "<coarse match length> <coarse match threshold>\n", argv[0]);
    return 1;
  }

  const int kBatchSize = 1024;
  char seq[] = {'A', 'C', 'T', 'G'};
  const int tseq_size = atoi(argv[1]);
  const int qseq_size = atoi(argv[2]);
  const int coarse_match_length = atoi(argv[3]);
  const int coarse_match_threshold = atoi(argv[4]);
  
  std::vector<char> target_sequence(tseq_size);
  std::vector<char> query_sequence(qseq_size);

  srand(123);
  for (int i = 0; i < tseq_size; i++) target_sequence[i] = seq[rand()%4];
  for (int i = 0; i < qseq_size; i++) query_sequence[i] = seq[rand()%4];

  char *d_target, *d_query, *d_batch_result;
  cudaMalloc((void**)&d_target, tseq_size * sizeof(char));
  cudaMalloc((void**)&d_query, qseq_size * sizeof(char));
  cudaMalloc((void**)&d_batch_result, kBatchSize * sizeof(char));

  cudaMemcpy(d_target, target_sequence.data(), tseq_size * sizeof(char), cudaMemcpyHostToDevice);
  cudaMemcpy(d_query, query_sequence.data(), qseq_size * sizeof(char), cudaMemcpyHostToDevice);

  uint32_t max_searchable_length = tseq_size - coarse_match_length;
  uint32_t current_position = 0;

  // host and device results
  char batch_result[kBatchSize];
  char batch_result_ref[kBatchSize];

  float total_time = 0.f;

  int error = 0;
  while (current_position < max_searchable_length) {
    cudaMemset(d_batch_result, 0, kBatchSize);
    memset(batch_result_ref, 0, kBatchSize);

    uint32_t end_position = current_position + kBatchSize;
    if (end_position >= max_searchable_length) {
      end_position = max_searchable_length;
    }
    uint32_t length = end_position - current_position;

    dim3 block_size(256);
    dim3 grid_size((length + block_size.x - 1) / block_size.x);

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    ga<<<grid_size, block_size>>>(
        d_target, d_query, d_batch_result, length, qseq_size,
        coarse_match_length, coarse_match_threshold, current_position);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;

    reference(target_sequence.data(), query_sequence.data(), batch_result_ref, length, qseq_size,
              coarse_match_length, coarse_match_threshold, current_position);

    cudaMemcpy(batch_result, d_batch_result, kBatchSize * sizeof(char), cudaMemcpyDeviceToHost);

    error = memcmp(batch_result_ref, batch_result, kBatchSize * sizeof(char));
    if (error) break;

    current_position = end_position;
  }
  printf("Total kernel execution time %f (s)\n", total_time * 1e-9f);
  printf("%s\n", error ? "FAIL" : "PASS");
  if (error) exit(1);
  
  cudaFree(d_target);
  cudaFree(d_query);
  cudaFree(d_batch_result);
  return 0;
}
