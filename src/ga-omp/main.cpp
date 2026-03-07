#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <omp.h>
#include "reference.h"

void ga(const char *__restrict target,
        const char *__restrict query,
              char *__restrict batch_result,
              uint32_t length,
              int query_sequence_length,
              int coarse_match_length,
              int coarse_match_threshold,
              int current_position)
{
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (uint tid = 0; tid < length; tid++) { 
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

  char *d_target = target_sequence.data(); 
  char *d_query = query_sequence.data();

  uint32_t max_searchable_length = tseq_size - coarse_match_length;
  uint32_t current_position = 0;

  // host and device results
  char d_batch_result[kBatchSize];
  char batch_result_ref[kBatchSize];

  float total_time = 0.f;

  int error = 0;

  #pragma omp target data map (to: d_target[0:tseq_size], \
                                   d_query[0:qseq_size]) \
                          map (alloc: d_batch_result[0:kBatchSize])
  {
    while (current_position < max_searchable_length) {
      #pragma omp target teams distribute parallel for thread_limit(256)
      for (int i = 0; i < kBatchSize; i++)
        d_batch_result[i] = 0;
      
      memset(batch_result_ref, 0, kBatchSize);

      uint32_t end_position = current_position + kBatchSize;
      if (end_position >= max_searchable_length) {
        end_position = max_searchable_length;
      }
      uint32_t length = end_position - current_position;

      auto start = std::chrono::steady_clock::now();

      ga( d_target, d_query, d_batch_result, length, qseq_size,
          coarse_match_length, coarse_match_threshold, current_position);

      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      total_time += time;

      reference(target_sequence.data(), query_sequence.data(), batch_result_ref, length, qseq_size,
                coarse_match_length, coarse_match_threshold, current_position);

      #pragma omp target update from (d_batch_result[0:kBatchSize])
      error = memcmp(batch_result_ref, d_batch_result, kBatchSize * sizeof(char));
      if (error) break;

      current_position = end_position;
    }
  }
  printf("Total kernel execution time %f (s)\n", total_time * 1e-9f);
  printf("%s\n", error ? "FAIL" : "PASS");
  if (error) exit(1);
  return 0;
}
