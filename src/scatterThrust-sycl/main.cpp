#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <cstdlib>

// Reference: /opt/intel/oneapi/dpcpp-ct/
template <typename Policy, typename InputIter1, typename InputIter2,
          typename OutputIter>
void dpct_scatter(Policy &&policy, InputIter1 first, InputIter1 last, InputIter2 map,
                  OutputIter result) {
  static_assert(
      std::is_same<typename std::iterator_traits<InputIter1>::iterator_category,
                   std::random_access_iterator_tag>::value &&
          std::is_same<
              typename std::iterator_traits<InputIter2>::iterator_category,
              std::random_access_iterator_tag>::value &&
          std::is_same<
              typename std::iterator_traits<OutputIter>::iterator_category,
              std::random_access_iterator_tag>::value,
      "Iterators passed to algorithms must be random-access iterators.");
  oneapi::dpl::copy(policy, first, last,
                    oneapi::dpl::make_permutation_iterator(result, map));
}

template <typename scalar_t> void scatter(sycl::queue &q, int64_t num_elems, int repeat) {
  int64_t src_size_bytes = num_elems * sizeof(scalar_t);
  int64_t idx_size_bytes = num_elems * sizeof(int64_t);
  int64_t out_size_bytes = num_elems * sizeof(scalar_t);

  int64_t *h_idx = (int64_t*) malloc (idx_size_bytes);
  scalar_t *h_src = (scalar_t*) malloc (src_size_bytes);
  scalar_t *h_out = (scalar_t*) malloc (out_size_bytes);
  srand(123);
  for (int64_t i = 0; i < num_elems; i++) {
    h_idx[i] = num_elems - 1 - i;
    h_src[i] = i;
  }

  scalar_t *d_src, *d_out;
  int64_t *d_idx;
  d_src = (scalar_t *)sycl::malloc_device(src_size_bytes, q);
  d_out = (scalar_t *)sycl::malloc_device(out_size_bytes, q);
  d_idx = (int64_t *)sycl::malloc_device(idx_size_bytes, q);
  q.memcpy(d_idx, h_idx, idx_size_bytes);
  q.memcpy(d_src, h_src, src_size_bytes);

  int64_t time = 0;
  for (int i = 0; i < repeat; i++) {
    q.memset(d_out, 0, out_size_bytes);
    q.wait();
    auto start = std::chrono::steady_clock::now();
    auto policy = oneapi::dpl::execution::make_device_policy(q);
    dpct_scatter(policy, d_src, d_src + num_elems, d_idx, d_out);
    q.wait();
    auto end = std::chrono::steady_clock::now();
    time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  }
  printf("Average execution time of scatter: %f (us)\n", (time * 1e-3f) / repeat);

  q.memcpy(h_out, d_out, out_size_bytes).wait();

  bool ok = true;
  for (int64_t i = 0; i < num_elems; i++) {
    if (h_out[i] != scalar_t(num_elems - 1 - i)) {
      ok = false;
      break;
    }
  }
  printf("%s\n\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  sycl::free(d_src, q);
  sycl::free(d_idx, q);
  sycl::free(d_out, q);
  free(h_src);
  free(h_idx);
  free(h_out);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int64_t num_elements = atol(argv[1]);
  const int repeat = atoi(argv[2]);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  printf("INT8 scatter\n");
  scatter<int8_t>(q, num_elements, repeat);
  printf("INT16 scatter\n");
  scatter<int16_t>(q, num_elements, repeat);
  printf("INT32 scatter\n");
  scatter<int32_t>(q, num_elements, repeat);
  printf("INT64 scatter\n");
  scatter<int64_t>(q, num_elements, repeat);
  printf("FP32 scatter\n");
  scatter<float>(q, num_elements, repeat);
  printf("FP64 scatter\n");
  scatter<double>(q, num_elements, repeat);
  return 0;
}
