#include <iostream>
#include <chrono>
#include <cstring>
#include <sycl/sycl.hpp>

template <typename T, int VEC_SIZE>
struct alignas(sizeof(T) * VEC_SIZE) data_t {
    T data[VEC_SIZE];
};

template <typename T, int vec_size>
void threads_copy_kernel(sycl::nd_item<3> &item, const T *in, T *out, const size_t n) {
    const int block_work_size = item.get_local_range(2) * vec_size;
    // get_group(dim) returns size_t
    auto index = item.get_group(2) * block_work_size +
                 item.get_local_id(2) * vec_size;
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

template <int vec_size, typename scalar_t>
void test_threads_copy(sycl::queue &q, size_t n, int repeat) try {
    auto in_h = new scalar_t[n];
    auto out_h = new scalar_t[n];
    for (size_t i = 0; i < n; i++)
        in_h[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);

    scalar_t *in_d, *out_d;
    in_d = sycl::malloc_device<scalar_t>(n, q);
    out_d = sycl::malloc_device<scalar_t>(n, q);
    q.memcpy(in_d, in_h, n * sizeof(scalar_t)).wait();

    const int block_size = 256;
    const int block_work_size = block_size * vec_size;

    sycl::range<3> lws (1, 1, block_size);
    sycl::range<3> gws (1, 1, (n + block_work_size - 1) / block_work_size * block_size);

    // warmup and verify
    for (int i = 0; i < 100; i++)
    {
        q.parallel_for(
            sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> item) {
                threads_copy_kernel<scalar_t, vec_size>(item, in_d, out_d, n);
        });
    }
    q.memcpy(out_h, out_d, n * sizeof(scalar_t)).wait();

    int s = memcmp(out_h, in_h, sizeof(scalar_t) * n);
    std::cout << (s ? "FAIL" : "PASS") << std::endl;

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
    {
        q.parallel_for(
            sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> item) {
                threads_copy_kernel<scalar_t, vec_size>(item, in_d, out_d, n);
        });
    }
    q.wait();

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    float avg_time_ms = time * 1e-6 / repeat;
    std::cout << "Average kernel execution time (ms):" << avg_time_ms << " Throughput:";

    // read + write
    float total_GBytes = 2 * n * sizeof(scalar_t) / 1000.0 / 1000.0;
    std::cout << total_GBytes / (avg_time_ms) << " GB/s" << std::endl;

    sycl::free(in_d, q);
    sycl::free(out_d, q);
    delete[] in_h;
    delete[] out_h;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
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

#ifdef USE_GPU
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
    sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

    std::cout << "int1: ";
    test_threads_copy<1, int>(q, numel, repeat);
    std::cout << "int2: ";
    test_threads_copy<2, int>(q, numel, repeat);
    std::cout << "int4: ";
    test_threads_copy<4, int>(q, numel, repeat);
    std::cout << "int8: ";
    test_threads_copy<8, int>(q, numel, repeat);
    std::cout << "int16: ";
    test_threads_copy<16, int>(q, numel, repeat);

    std::cout << "short1: ";
    test_threads_copy<1, short>(q, numel * 2, repeat);
    std::cout << "short2: ";
    test_threads_copy<2, short>(q, numel * 2, repeat);
    std::cout << "short4: ";
    test_threads_copy<4, short>(q, numel * 2, repeat);
    std::cout << "short8: ";
    test_threads_copy<8, short>(q, numel * 2, repeat);
    std::cout << "short16: ";
    test_threads_copy<16, short>(q, numel * 2, repeat);

    std::cout << "char1: ";
    test_threads_copy<1, char>(q, numel * 4, repeat);
    std::cout << "char2: ";
    test_threads_copy<2, char>(q, numel * 4, repeat);
    std::cout << "char4: ";
    test_threads_copy<4, char>(q, numel * 4, repeat);
    std::cout << "char8: ";
    test_threads_copy<8, char>(q, numel * 4, repeat);
    std::cout << "char16: ";
    test_threads_copy<16, char>(q, numel * 4, repeat);
    return 0;
}
