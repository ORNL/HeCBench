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
    if (index >= n) return;
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
bool test_threads_copy(sycl::queue &q, size_t n, int repeat) try {
    const int block_size = 256;
    const int block_work_size = block_size * vec_size;
    const size_t num_blocks = (n + block_work_size - 1) / block_work_size;

    // Pad the buffers to the whole launch grid so that a store past n lands
    // in the padding, where it is detected, instead of outside the allocation.
    const size_t alloc_n = num_blocks * block_work_size;

    auto in_h = new scalar_t[alloc_n];
    auto out_h = new scalar_t[alloc_n];
    for (size_t i = 0; i < n; i++)
        in_h[i] = static_cast<scalar_t>(rand() % 127);
    for (size_t i = n; i < alloc_n; i++)
        in_h[i] = static_cast<scalar_t>(127);

    scalar_t *in_d, *out_d;
    in_d = sycl::malloc_device<scalar_t>(alloc_n, q);
    out_d = sycl::malloc_device<scalar_t>(alloc_n, q);
    q.memcpy(in_d, in_h, alloc_n * sizeof(scalar_t));
    q.memset(out_d, 0xFF, alloc_n * sizeof(scalar_t)).wait();

    sycl::range<3> lws (1, 1, block_size);
    sycl::range<3> gws (1, 1, num_blocks * block_size);

    auto thrcpyFn = [=](sycl::nd_item<3> item) {
         threads_copy_kernel<scalar_t, vec_size>(item, in_d, out_d, n);
    };

    // warmup and verify
    for (int i = 0; i < 100; i++)
    {
        q.parallel_for(sycl::nd_range<3>(gws, lws), thrcpyFn);
    }
    q.memcpy(out_h, out_d, alloc_n * sizeof(scalar_t)).wait();

    bool ok = memcmp(out_h, in_h, sizeof(scalar_t) * n) == 0;
    for (size_t i = n; i < alloc_n; i++) {
        if (out_h[i] != static_cast<scalar_t>(-1)) {
            ok = false;
            break;
        }
    }
    std::cout << (ok ? "PASS" : "FAIL") << std::endl;

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
    {
        q.parallel_for(sycl::nd_range<3>(gws, lws), thrcpyFn);
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
    return ok;
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
    bool ok = true;

    std::cout << "1GB threads copy test ..." << std::endl;
    const size_t numel = 1024 * 1024 * 256 + 2;
    srand(19937);

#ifdef USE_GPU
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
    sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

    std::cout << "int1: ";
    ok &= test_threads_copy<1, int>(q, numel, repeat);
    std::cout << "int2: ";
    ok &= test_threads_copy<2, int>(q, numel, repeat);
    std::cout << "int4: ";
    ok &= test_threads_copy<4, int>(q, numel, repeat);
    std::cout << "int8: ";
    ok &= test_threads_copy<8, int>(q, numel, repeat);
    std::cout << "int16: ";
    ok &= test_threads_copy<16, int>(q, numel, repeat);

    std::cout << "short1: ";
    ok &= test_threads_copy<1, short>(q, numel * 2, repeat);
    std::cout << "short2: ";
    ok &= test_threads_copy<2, short>(q, numel * 2, repeat);
    std::cout << "short4: ";
    ok &= test_threads_copy<4, short>(q, numel * 2, repeat);
    std::cout << "short8: ";
    ok &= test_threads_copy<8, short>(q, numel * 2, repeat);
    std::cout << "short16: ";
    ok &= test_threads_copy<16, short>(q, numel * 2, repeat);

    std::cout << "char1: ";
    ok &= test_threads_copy<1, char>(q, numel * 4, repeat);
    std::cout << "char2: ";
    ok &= test_threads_copy<2, char>(q, numel * 4, repeat);
    std::cout << "char4: ";
    ok &= test_threads_copy<4, char>(q, numel * 4, repeat);
    std::cout << "char8: ";
    ok &= test_threads_copy<8, char>(q, numel * 4, repeat);
    std::cout << "char16: ";
    ok &= test_threads_copy<16, char>(q, numel * 4, repeat);
    return ok ? 0 : 1;
}
