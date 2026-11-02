#include <iostream>
#include <chrono>
#include <cstring>
#include <hip/hip_runtime.h>

#define GPU_CHECK(x) do { \
  hipError_t err = x; \
  if (err != hipSuccess) { \
    printf("HIP error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
    exit(1); \
  } \
} while (0)


template <typename T, int VEC_SIZE>
struct alignas(sizeof(T) * VEC_SIZE) data_t {
    T data[VEC_SIZE];
};

template <typename T, int vec_size>
__global__ void threads_copy_kernel(const T *in, T *out, const size_t n) {
    const int block_work_size = blockDim.x * vec_size;
    auto index = static_cast<size_t>(blockIdx.x) * block_work_size + threadIdx.x * vec_size;
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
bool test_threads_copy(size_t n, int repeat) {
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
    GPU_CHECK(hipMalloc(&in_d, alloc_n * sizeof(scalar_t)));
    GPU_CHECK(hipMalloc(&out_d, alloc_n * sizeof(scalar_t)));
    GPU_CHECK(hipMemcpy(in_d, in_h, alloc_n * sizeof(scalar_t), hipMemcpyHostToDevice));
    GPU_CHECK(hipMemset(out_d, 0xFF, alloc_n * sizeof(scalar_t)));

    dim3 threadsPerBlock(block_size);
    dim3 numBlocks(num_blocks);

    // warmup and verify
    for (int i = 0; i < 100; i++)
        threads_copy_kernel<scalar_t, vec_size><<<numBlocks, threadsPerBlock>>>(in_d, out_d, n);
    GPU_CHECK(hipMemcpy(out_h, out_d, alloc_n * sizeof(scalar_t), hipMemcpyDeviceToHost));

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
        threads_copy_kernel<scalar_t, vec_size><<<numBlocks, threadsPerBlock>>>(in_d, out_d, n);
    GPU_CHECK(hipDeviceSynchronize());

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    float avg_time_ms = time * 1e-6 / repeat;
    std::cout << "Average kernel execution time (ms):" << avg_time_ms << " Throughput:";

    // read + write
    float total_GBytes = 2 * n * sizeof(scalar_t) / 1000.0 / 1000.0;
    std::cout << total_GBytes / (avg_time_ms) << " GB/s" << std::endl;

    GPU_CHECK(hipFree(in_d));
    GPU_CHECK(hipFree(out_d));
    delete[] in_h;
    delete[] out_h;
    return ok;
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

    std::cout << "int1: ";
    ok &= test_threads_copy<1, int>(numel, repeat);
    std::cout << "int2: ";
    ok &= test_threads_copy<2, int>(numel, repeat);
    std::cout << "int4: ";
    ok &= test_threads_copy<4, int>(numel, repeat);
    std::cout << "int8: ";
    ok &= test_threads_copy<8, int>(numel, repeat);
    std::cout << "int16: ";
    ok &= test_threads_copy<16, int>(numel, repeat);

    std::cout << "short1: ";
    ok &= test_threads_copy<1, short>(numel * 2, repeat);
    std::cout << "short2: ";
    ok &= test_threads_copy<2, short>(numel * 2, repeat);
    std::cout << "short4: ";
    ok &= test_threads_copy<4, short>(numel * 2, repeat);
    std::cout << "short8: ";
    ok &= test_threads_copy<8, short>(numel * 2, repeat);
    std::cout << "short16: ";
    ok &= test_threads_copy<16, short>(numel * 2, repeat);

    std::cout << "char1: ";
    ok &= test_threads_copy<1, char>(numel * 4, repeat);
    std::cout << "char2: ";
    ok &= test_threads_copy<2, char>(numel * 4, repeat);
    std::cout << "char4: ";
    ok &= test_threads_copy<4, char>(numel * 4, repeat);
    std::cout << "char8: ";
    ok &= test_threads_copy<8, char>(numel * 4, repeat);
    std::cout << "char16: ";
    ok &= test_threads_copy<16, char>(numel * 4, repeat);
    return ok ? 0 : 1;
}
