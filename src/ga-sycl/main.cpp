#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include <sycl/sycl.hpp>
#include "reference.h"

#ifdef __NVPTX__
  #include <sycl/ext/oneapi/experimental/cuda/builtins.hpp>
  using namespace sycl::ext::oneapi::experimental::cuda;
#else
  #define ldg(a) (*(a))
#endif

void ga(sycl::nd_item<1> &item,
        const char *__restrict target,
        const char *__restrict query,
              char *__restrict batch_result,
              uint32_t length,
              int query_sequence_length,
              int coarse_match_length,
              int coarse_match_threshold,
              int current_position)
{
  uint tid = item.get_global_id(0);
  if (tid > length) return;
  bool match = false;
  const int max_length = query_sequence_length - coarse_match_length;
  const int base = current_position + tid;

  for (int i = 0; i <= max_length; i++) {
    int distance = 0;
    for (int j = 0; j < coarse_match_length; j++) {
      if (ldg(&target[base + j]) != ldg(&query[i + j])) {
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

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  char *d_target = sycl::malloc_device<char>(tseq_size, q);
  char *d_query = sycl::malloc_device<char>(qseq_size, q);
  char *d_batch_result = sycl::malloc_device<char>(kBatchSize, q);

  q.memcpy(d_target, target_sequence.data(), tseq_size * sizeof(char));
  q.memcpy(d_query, query_sequence.data(), qseq_size * sizeof(char));

  uint32_t max_searchable_length = tseq_size - coarse_match_length;
  uint32_t current_position = 0;

  // host and device results
  char batch_result[kBatchSize];
  char batch_result_ref[kBatchSize];

  float total_time = 0.f;

  int error = 0;
  while (current_position < max_searchable_length) {
    q.memset(d_batch_result, 0, kBatchSize);
    memset(batch_result_ref, 0, kBatchSize);

    uint32_t end_position = current_position + kBatchSize;
    if (end_position >= max_searchable_length) {
      end_position = max_searchable_length;
    }
    uint32_t length = end_position - current_position;

    sycl::range<1> lws (256);
    sycl::range<1> gws ((length + 255) / 256 * 256);

    q.wait();
    auto start = std::chrono::steady_clock::now();

    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class genetic>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        ga (item, d_target, d_query, d_batch_result,
            length, qseq_size, coarse_match_length,
            coarse_match_threshold, current_position);
      });
    }).wait();

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;

    reference(target_sequence.data(), query_sequence.data(), batch_result_ref, length, qseq_size,
              coarse_match_length, coarse_match_threshold, current_position);

    q.memcpy(batch_result, d_batch_result, kBatchSize * sizeof(char)).wait();

    error = memcmp(batch_result_ref, batch_result, kBatchSize * sizeof(char));
    if (error) break;

    current_position = end_position;
  }
  printf("Total kernel execution time %f (s)\n", total_time * 1e-9f);
  printf("%s\n", error ? "FAIL" : "PASS");
  if (error) exit(1);

  sycl::free(d_target, q);
  sycl::free(d_query, q);
  sycl::free(d_batch_result, q);
  return 0;
}
