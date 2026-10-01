#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cub/cub.cuh>
#include "zipf.h"

#define CHECK(call)                                                          \
  do {                                                                       \
    const cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                                \
      fprintf(stderr, "CUDA error %s:%d '%s': %s\n", __FILE__, __LINE__,     \
              #call, cudaGetErrorString(err));                               \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

static_assert(sizeof(offset_t) == 8, "64-bit CUB offset iterators required");

template <typename Op>
void bench_op(const char *name, Op op, value_t *d_in, value_t *d_out,
              offset_t *d_offsets, offset_t num_segments, offset_t num_items,
              int repeat, const std::vector<value_t> &ref)
{
  void *d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
  CHECK(op(d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments, d_offsets));
  if (temp_storage_bytes > 0)
    CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  CHECK(op(d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments, d_offsets));
  CHECK(cudaDeviceSynchronize());

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++)
    CHECK(op(d_temp_storage, temp_storage_bytes, d_in, d_out, num_segments, d_offsets));
  CHECK(cudaDeviceSynchronize());
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("%s throughput = %f (G/s)\n", name, 1.f * num_items * repeat / time);

  std::vector<value_t> h_out(num_segments);
  CHECK(cudaMemcpy(h_out.data(), d_out, num_segments * sizeof(value_t),
                   cudaMemcpyDeviceToHost));
  count_errors(h_out, ref, name);

  if (d_temp_storage)
    CHECK(cudaFree(d_temp_storage));
}

void segreduce(offset_t num_items, int repeat)
{
  SegProblem p = make_zipf_problem(num_items);
  print_problem(p);

  std::vector<value_t> ref_sum, ref_min, ref_max;
  cpu_segmented(p, ref_sum, ref_min, ref_max);

  value_t *d_in, *d_out;
  offset_t *d_offsets;
  CHECK(cudaMalloc(&d_in, num_items * sizeof(value_t)));
  CHECK(cudaMalloc(&d_out, p.num_segments * sizeof(value_t)));
  CHECK(cudaMalloc(&d_offsets, (p.num_segments + 1) * sizeof(offset_t)));
  CHECK(cudaMemcpy(d_in, p.values.data(), num_items * sizeof(value_t),
                   cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_offsets, p.offsets.data(),
                   (p.num_segments + 1) * sizeof(offset_t),
                   cudaMemcpyHostToDevice));

  // CUB 64-bit DeviceSegmentedReduce: int64_t num_segments and 64-bit offsets.
  auto sum_op = [](void *tmp, size_t &bytes, value_t *in, value_t *out,
                   offset_t n, offset_t *off) {
    return cub::DeviceSegmentedReduce::Sum(tmp, bytes, in, out, n, off, off + 1);
  };
  auto min_op = [](void *tmp, size_t &bytes, value_t *in, value_t *out,
                   offset_t n, offset_t *off) {
    return cub::DeviceSegmentedReduce::Min(tmp, bytes, in, out, n, off, off + 1);
  };
  auto max_op = [](void *tmp, size_t &bytes, value_t *in, value_t *out,
                   offset_t n, offset_t *off) {
    return cub::DeviceSegmentedReduce::Max(tmp, bytes, in, out, n, off, off + 1);
  };

  bench_op("cub::DeviceSegmentedReduce::Sum", sum_op, d_in, d_out, d_offsets,
           p.num_segments, num_items, repeat, ref_sum);
  bench_op("cub::DeviceSegmentedReduce::Min", min_op, d_in, d_out, d_offsets,
           p.num_segments, num_items, repeat, ref_min);
  bench_op("cub::DeviceSegmentedReduce::Max", max_op, d_in, d_out, d_offsets,
           p.num_segments, num_items, repeat, ref_max);

  CHECK(cudaFree(d_in));
  CHECK(cudaFree(d_out));
  CHECK(cudaFree(d_offsets));
}

int main(int argc, char *argv[])
{
  if (argc != 3) {
    printf("Usage: %s <multiplier> <repeat>\n", argv[0]);
    printf("The total number of elements is 16384 x multiplier\n");
    return 1;
  }
  const int multiplier = atoi(argv[1]);
  const int repeat = atoi(argv[2]);
  const offset_t num_items = 16384 * offset_t(multiplier);
  if (num_items <= 0)
    return 0;
  segreduce(num_items, repeat);
  return 0;
}
