#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/numeric>
#include <sycl/sycl.hpp>
#include <algorithm>  // shuffle
#include <cstdio>
#include <chrono>
#include <numeric> // iota
#include <random>
#include <vector>
#include <cstdlib>

template <typename T>
void sort_key_value (sycl::queue &q, int n, int repeat, bool verify) {

  printf("Number of keys is %d and the size of each value in bytes is %zu\n", n, sizeof(T));

  unsigned seed = 123;
  bool ok = true;

  std::vector<int> keys (n);
  std::vector<T> vals (n);
  std::iota(keys.begin(), keys.end(), 0);

  auto policy = oneapi::dpl::execution::make_device_policy(q);

  double total_time = 0.0;
  for (int i = 0; i < repeat; i++) {
    // initialize keys and values
    std::shuffle(keys.begin(), keys.end(), std::default_random_engine(seed));
    for (int i = 0; i < n; i++) vals[i] = keys[i] % 256;

    auto start = std::chrono::steady_clock::now();

    auto zipped_begin = oneapi::dpl::make_zip_iterator(keys.begin(), vals.begin());
    std::stable_sort(policy, zipped_begin, zipped_begin + n,
      [](auto lhs, auto rhs) { return std::get<0>(lhs) < std::get<0>(rhs); });

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;
  }

  if (!verify) 
    printf("Average sort time %f (us)\n", (total_time * 1e-3) / repeat);
  else {
    for (int i = 0; i < n; i++) {
      if (keys[i] != i || vals[i] != i % 256) {
        ok = false;
        break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
    if (!ok) exit(1);
  }
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <number of keys> <repeat>\n", argv[0]);
    return 1;
  }
  const int size = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  printf("\nWarmup and verify\n");
  sort_key_value<unsigned char>(q, size, repeat, true);
  sort_key_value<short>(q, size, repeat, true);
  sort_key_value<int>(q, size, repeat, true);
  sort_key_value<long>(q, size, repeat, true);

  printf("\nPerformance evaluation\n");
  sort_key_value<unsigned char>(q, size, repeat, false);
  sort_key_value<short>(q, size, repeat, false);
  sort_key_value<int>(q, size, repeat, false);
  sort_key_value<long>(q, size, repeat, false);

  return 0;
}
