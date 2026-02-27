#include <iostream>
#include <chrono>
#include <cstring>
#include <omp.h>

template <typename T, int VEC_SIZE>
struct alignas(sizeof(T) * VEC_SIZE) data_t {
    T data[VEC_SIZE];
};

// baseline
template <typename T>
void threads_copy_kernel_base(const T *in, T *out, const size_t n) {
  #pragma omp target teams distribute parallel for
  for (auto i = 0; i < n; i++) {
    out[i] = in[i];
  }
}

template <typename T, int vec_size>
void threads_copy_kernel(const T *in, T *out, const size_t n, const unsigned numTeams) {
  const int block_work_size = 256 * vec_size;
  #pragma omp target teams distribute parallel for collapse(2)
  for (auto n = 0; n < numTeams; n++) {
    for (auto t = 0; t < 256; t++) {
      auto index = static_cast<size_t>(n) * block_work_size + t * vec_size;
      auto remaining = n - index;
      if (remaining < vec_size) {
          for (auto i = index; i < n; i++) {
              out[i] = in[i];
          }
      } else {
          using vec_t = data_t<T, vec_size>;
          auto in_vec = reinterpret_cast<const vec_t *>(in + index);
          auto out_vec = reinterpret_cast<vec_t *>(out + index);
          *out_vec = *in_vec;
      }
    }
  }
}

template <int vec_size, typename scalar_t>
void test_threads_copy(size_t n, int repeat) {
    auto in = new scalar_t[n];
    auto out = new scalar_t[n];
    for (size_t i = 0; i < n; i++)
        in[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    const int block_size = 256;
    const int block_work_size = block_size * vec_size;

    unsigned numTeams = ((n + block_work_size - 1) / block_work_size);

    #pragma omp target data map(to: in[0:n]) map(alloc: out[0:n])
    {
      // warmup and verify
      for (int i = 0; i < 100; i++)
          threads_copy_kernel<scalar_t, vec_size>(in, out, n, numTeams);
      #pragma omp target update from (out[0:n]) 

      int s = memcmp(out, in, sizeof(scalar_t) * n);
      std::cout << (s ? "FAIL" : "PASS") << std::endl;

      auto start = std::chrono::steady_clock::now();

      for (int i = 0; i < repeat; i++)
          threads_copy_kernel_base<scalar_t>(in, out, n);

      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      float avg_time_ms = time * 1e-6 / repeat;
      std::cout << "Average base kernel execution time (ms):" << avg_time_ms << " Throughput:";

      // read + write
      float total_GBytes = 2 * n * sizeof(scalar_t) / 1000.0 / 1000.0;
      std::cout << total_GBytes / (avg_time_ms) << " GB/s" << std::endl;

      start = std::chrono::steady_clock::now();

      for (int i = 0; i < repeat; i++)
          threads_copy_kernel<scalar_t, vec_size>(in, out, n, numTeams);

      end = std::chrono::steady_clock::now();
      time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      avg_time_ms = time * 1e-6 / repeat;
      std::cout << "Average kernel execution time (ms):" << avg_time_ms << " Throughput:";
      std::cout << total_GBytes / (avg_time_ms) << " GB/s" << std::endl;
    }

    delete[] in;
    delete[] out;
}

int main(int argc, char* argv[])
{
    if (argc != 2) {
      std::cout << "Usage: " << argv[0] << " <repeat>" << std::endl;
      return 1;
    }
    const int repeat = atoi(argv[1]);

    std::cout << "1GB threads copy test ..." << std::endl;
    const size_t numel = 1024 * 1024 * 256 + 2;
    srand(19937);

    std::cout << "int1: ";
    test_threads_copy<1, int>(numel, repeat);
    std::cout << "int2: ";
    test_threads_copy<2, int>(numel, repeat);
    std::cout << "int4: ";
    test_threads_copy<4, int>(numel, repeat);
    std::cout << "int8: ";
    test_threads_copy<8, int>(numel, repeat);
    std::cout << "int16: ";
    test_threads_copy<16, int>(numel, repeat);

    std::cout << "short1: ";
    test_threads_copy<1, short>(numel * 2, repeat);
    std::cout << "short2: ";
    test_threads_copy<2, short>(numel * 2, repeat);
    std::cout << "short4: ";
    test_threads_copy<4, short>(numel * 2, repeat);
    std::cout << "short8: ";
    test_threads_copy<8, short>(numel * 2, repeat);
    std::cout << "short16: ";
    test_threads_copy<16, short>(numel * 2, repeat);

    std::cout << "char1: ";
    test_threads_copy<1, char>(numel * 4, repeat);
    std::cout << "char2: ";
    test_threads_copy<2, char>(numel * 4, repeat);
    std::cout << "char4: ";
    test_threads_copy<4, char>(numel * 4, repeat);
    std::cout << "char8: ";
    test_threads_copy<8, char>(numel * 4, repeat);
    std::cout << "char16: ";
    test_threads_copy<16, char>(numel * 4, repeat);
    return 0;
}
