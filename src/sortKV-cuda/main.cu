#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
#include <thrust/sequence.h>
#include <thrust/shuffle.h>
#include <thrust/random.h>
#include <cstdio>
#include <chrono>
#include <vector>
#include <cstdlib>

template <typename T>
void sort_key_value (int n, int repeat, bool verify) {

  printf("Number of keys is %d and the size of each value in bytes is %zu\n", n, sizeof(T));

  unsigned seed = 123;
  bool ok = true;

  thrust::host_vector<int> keys(n);
  thrust::host_vector<T> vals(n);
  thrust::sequence(keys.begin(), keys.end());

  thrust::device_vector<int> d_keys;
  thrust::device_vector<T> d_vals;

  double total_time = 0.0;
  for (int i = 0; i < repeat; i++) {
    // initialize keys and values
    thrust::shuffle(keys.begin(), keys.end(), thrust::default_random_engine(seed));
    for (int i = 0; i < n; i++) vals[i] = keys[i] % 256;

    auto start = std::chrono::steady_clock::now();

    d_keys = keys;
    d_vals = vals;

    thrust::sort_by_key(d_keys.begin(), d_keys.end(), d_vals.begin());

    keys = d_keys;
    vals = d_vals;

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

  printf("\nWarmup and verify\n");
  sort_key_value<unsigned char>(size, repeat, true);
  sort_key_value<short>(size, repeat, true);
  sort_key_value<int>(size, repeat, true);
  sort_key_value<long>(size, repeat, true);

  printf("\nPerformance evaluation\n");
  sort_key_value<unsigned char>(size, repeat, false);
  sort_key_value<short>(size, repeat, false);
  sort_key_value<int>(size, repeat, false);
  sort_key_value<long>(size, repeat, false);

  return 0;
}
