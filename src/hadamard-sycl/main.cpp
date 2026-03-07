#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <sycl/sycl.hpp>

#include "utils.h"
#include "specials.h"
#include "reference.h"

struct HadamardParamsBase {
    using index_t = int64_t;

    int batch, dim, log_N;

    index_t x_batch_stride;
    index_t out_batch_stride;

    float scale;

    // Common data pointers.
    void *__restrict__ x_ptr;
    void *__restrict__ out_ptr;
};

void set_hadamard_params(HadamardParamsBase &params,
                         // sizes
                         const size_t batch,
                         const size_t dim,
                         const size_t multiple,
                         // device pointers
                         void* x,
                         void* out,
                         float scale
                         )
{

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    params.batch = batch;
    params.dim = dim;
    params.log_N = int(ceil(std::log2(dim / multiple)));

    // Set the pointers and strides.
    params.x_ptr = x;
    params.out_ptr = out;

    // All stride are in elements, not bytes.
    params.x_batch_stride = dim;  // seq_len * dim
    params.out_batch_stride = dim;  // seq_len * dim

    params.scale = scale;
}

template<int subGroupSize_, int kNThreads_, int kLogN_, typename input_t_>
struct fast_hadamard_transform_kernel_traits {
    using input_t = input_t_;
    static constexpr int kNThreads = kNThreads_;
    static constexpr int kLogN = kLogN_;
    static constexpr int N = 1 << kLogN;
    static constexpr int kNBytes = sizeof(input_t);
    static_assert(kNBytes == 2 || kNBytes == 4);
    static constexpr int kNElts = kNBytes == 4 ? 4 : 8;
    // It's possible that we need to do 2 rounds of exchange if input_t is 16 bits
    // (since then we'd have 8 values of float, and each round we can exchange 4 floats).
    static constexpr int kNExchangePerVec = sizeof(float) / sizeof(input_t);
    using vec_t = typename BytesToType<kNBytes * kNElts>::Type;
    static constexpr int kNChunks = N / (kNElts * kNThreads);
    // We don't want to use more than 32 KB of shared memory.
    static constexpr int kSmemExchangeSize = std::min(N * 4, 32 * 1024);
    static constexpr int kNExchangeRounds = N * 4 / kSmemExchangeSize;
    static_assert(kNExchangeRounds * kSmemExchangeSize == N * 4);
    static constexpr int kSmemSize = kSmemExchangeSize;
    static constexpr int kSubGroupSize = subGroupSize_;
};

template <int kNChunks>
inline void hadamard_mult_thread_chunk_12(float x[kNChunks][12]) {
#pragma unroll
    for (int c = 0; c < kNChunks; ++c) { hadamard_mult_thread_12(x[c]); }
}

template <int kNChunks>
inline void hadamard_mult_thread_chunk_20(float x[kNChunks][20]) {
#pragma unroll
    for (int c = 0; c < kNChunks; ++c) { hadamard_mult_thread_20(x[c]); }
}

template <int kNChunks>
inline void hadamard_mult_thread_chunk_28(float x[kNChunks][28]) {
#pragma unroll
    for (int c = 0; c < kNChunks; ++c) { hadamard_mult_thread_28(x[c]); }
}

template <int kNChunks>
inline void hadamard_mult_thread_chunk_40(float x[kNChunks][40]) {
#pragma unroll
    for (int c = 0; c < kNChunks; ++c) { hadamard_mult_thread_40(x[c]); }
}

template <typename Ktraits>
void fast_hadamard_transform_kernel(HadamardParamsBase params,
                                    uint8_t *smem, sycl::nd_item<3> &item)
{
    constexpr int kNThreads = Ktraits::kNThreads;
    constexpr int kNElts = Ktraits::kNElts;
    constexpr int kNExchangePerVec = Ktraits::kNExchangePerVec;
    constexpr int kNExchangeRounds = Ktraits::kNExchangeRounds;
    constexpr int kNChunks = Ktraits::kNChunks;
    constexpr int kSubGroupSize = Ktraits::kSubGroupSize;
    using input_t = typename Ktraits::input_t;
    using vec_t = typename Ktraits::vec_t;

    constexpr int kLogNElts = cilog2(Ktraits::kNElts);
    static_assert(1 << kLogNElts == kNElts, "kNElts must be a power of 2");
    constexpr int kWarpSize = kNThreads < kSubGroupSize ? kNThreads : kSubGroupSize;
    constexpr int kLogWarpSize = cilog2(kWarpSize);
    static_assert(1 << kLogWarpSize == kWarpSize, "Warp size must be a power of 2");
    constexpr int kNWarps = kNThreads / kWarpSize;
    constexpr int kLogNWarps = cilog2(kNWarps);
    static_assert(1 << kLogNWarps == kNWarps, "kNWarps must be a power of 2");
    constexpr int kLoadsPerExchange = Ktraits::kSmemExchangeSize / (sizeof(vec_t) * kNThreads);
    static_assert(kLoadsPerExchange * sizeof(vec_t) * kNThreads == Ktraits::kSmemExchangeSize, "kSmemExchangeSize should be a power of 2");
    static_assert(kNExchangeRounds * kLoadsPerExchange * sizeof(vec_t) == kNChunks * kNElts * sizeof(float));

    constexpr int kChunksPerExchange = Ktraits::kSmemExchangeSize / (sizeof(vec_t) * kNExchangePerVec * kNThreads);
    static_assert(kChunksPerExchange * sizeof(vec_t) * kNExchangePerVec * kNThreads == Ktraits::kSmemExchangeSize);
    constexpr int kNExchanges = kNChunks / kChunksPerExchange;
    static_assert(kNExchanges * kChunksPerExchange == kNChunks);

    // Shared memory.
    vec_t *smem_exchange = reinterpret_cast<vec_t *>(smem);

    const int batch_id = item.get_group(2);
    input_t *x = reinterpret_cast<input_t *>(params.x_ptr) + batch_id * params.x_batch_stride;
    input_t *out = reinterpret_cast<input_t *>(params.out_ptr) + batch_id * params.out_batch_stride;

    float x_vals[kNChunks][kNElts];
    load_input<kNChunks, kNElts, input_t>(x, x_vals, params.dim, item);

    hadamard_mult_thread<kLogNElts, kNChunks>(x_vals);
    hadamard_mult_warp<kLogWarpSize, 0, kNChunks, kNElts>(x_vals, item);

    if constexpr (kNWarps > 1) {
        exchange_smem_pre<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps, true, vec_t>(x_vals, smem_exchange, item);
        hadamard_mult_warp<kLogNWarps, 0, kNChunks, kNElts>(x_vals, item);
        exchange_smem_pre<kNChunks, kChunksPerExchange, kNElts, kWarpSize, kNWarps, false, vec_t>(x_vals, smem_exchange, item);
    }

    if constexpr (kNChunks > 1) {
        float x_vals_transposed[kNElts][kNChunks];
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) { x_vals_transposed[i][c] = x_vals[c][i]; }
        }
        if constexpr (kNChunks == 12) {
            hadamard_mult_thread_chunk_12<kNElts>(x_vals_transposed);
        } else if constexpr (kNChunks == 20) {
            hadamard_mult_thread_chunk_20<kNElts>(x_vals_transposed);
        } else if constexpr (kNChunks == 28) {
            hadamard_mult_thread_chunk_28<kNElts>(x_vals_transposed);
        } else if constexpr (kNChunks == 40) {
            hadamard_mult_thread_chunk_40<kNElts>(x_vals_transposed);
        } else {
            constexpr int kLogNChunks = cilog2(kNChunks);
            static_assert(1 << kLogNChunks == kNChunks, "kNChunks must be a power of 2");
            hadamard_mult_thread<kLogNChunks, kNElts>(x_vals_transposed);
        }
        #pragma unroll
        for (int c = 0; c < kNChunks; ++c) {
            #pragma unroll
            for (int i = 0; i < kNElts; ++i) { x_vals[c][i] = x_vals_transposed[i][c]; }
        }
    }

    store_output<kNChunks, kNElts, input_t>(out, x_vals, params.dim, params.scale, item);
}

template <int subGroupSize, int kNThreads, int kLogN, typename input_t>
void fast_hadamard_transform_launch(HadamardParamsBase &params, sycl::queue& stream) {
    using Ktraits = fast_hadamard_transform_kernel_traits<subGroupSize, kNThreads, kLogN, input_t>;
    sycl::range<3> gws (1, 1, params.batch * Ktraits::kNThreads);
    sycl::range<3> lws (1, 1, Ktraits::kNThreads);
    constexpr int kSmemSize = Ktraits::kSmemSize;
    /*
    auto kernel = &fast_hadamard_transform_kernel<Ktraits>;
    if (kSmemSize >= 48 * 1024) {
        printf("Increase kSmem size to %d\n", kSmemSize);
        GPU_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSmemSize));
        }
    kernel<<<grid, Ktraits::kNThreads, kSmemSize, stream>>>(params);
    */
   
    stream.submit([&](sycl::handler &cgh) {
      sycl::local_accessor<uint8_t, 1> smem(sycl::range<1>(kSmemSize), cgh);

      cgh.parallel_for(sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> item) {
        fast_hadamard_transform_kernel<Ktraits>(
          params,
          smem.get_multi_ptr<sycl::access::decorated::no>().get(),
          item);
      });
    });
}

template <typename input_t, int subGroupSize>
void fast_hadamard_transform(HadamardParamsBase &params, sycl::queue& stream) {
    if (params.log_N == 3) {
        fast_hadamard_transform_launch<subGroupSize, 1, 3, input_t>(params, stream);
    } else if (params.log_N == 4) {
        fast_hadamard_transform_launch<subGroupSize, 2, 4, input_t>(params, stream);
    } else if (params.log_N == 5) {
        fast_hadamard_transform_launch<subGroupSize, 4, 5, input_t>(params, stream);
    } else if (params.log_N == 6) {
        fast_hadamard_transform_launch<subGroupSize, 8, 6, input_t>(params, stream);
    } else if (params.log_N == 7) {
        fast_hadamard_transform_launch<subGroupSize, 16, 7, input_t>(params, stream);
    } else if (params.log_N == 8) {
        fast_hadamard_transform_launch<subGroupSize, 32, 8, input_t>(params, stream);
    } else if (params.log_N == 9) {
        fast_hadamard_transform_launch<subGroupSize, 32, 9, input_t>(params, stream);
    } else if (params.log_N == 10) {
        fast_hadamard_transform_launch<subGroupSize, 128, 10, input_t>(params, stream);
    } else if (params.log_N == 11) {
        fast_hadamard_transform_launch<subGroupSize, 256, 11, input_t>(params, stream);
    } else if (params.log_N == 12) {
        fast_hadamard_transform_launch<subGroupSize, 256, 12, input_t>(params, stream);
    } else if (params.log_N == 13) {
        fast_hadamard_transform_launch<subGroupSize, 256, 13, input_t>(params, stream);
    } else if (params.log_N == 14) {
        fast_hadamard_transform_launch<subGroupSize, 256, 14, input_t>(params, stream);
    } else if (params.log_N == 15) {
        fast_hadamard_transform_launch<subGroupSize, 256, 15, input_t>(params, stream);
    }
}

template <typename T>
void hadamard_transform(sycl::queue &q, int batch_size, int dim, int repeat) try {

    assert(dim % (8) == 0);
    assert(dim <= 32768);

    float scale = 1.f / std::sqrt(dim);
    int64_t numel = (int64_t)batch_size * dim;

    std::vector<T> h_x (numel);
    std::vector<T> h_out (numel);
    std::vector<T> r_out (numel);

    std::mt19937 gen(19937);
    std::uniform_real_distribution<float> dis(0.0, 1.0);

    for (int64_t i = 0; i < numel; ++i) {
      h_x[i] = T(dis(gen));
    }

    void *d_x, *d_out;
    d_x = (void *)sycl::malloc_device(numel * sizeof(T), q);
    d_out = (void *)sycl::malloc_device(numel * sizeof(T), q);
    q.memcpy(d_x, h_x.data(), numel * sizeof(T));

    HadamardParamsBase params;
    set_hadamard_params(params, batch_size, dim, 1, d_x, d_out, scale);

    auto sg_sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
    auto r = std::max_element(sg_sizes.begin(), sg_sizes.end());
    int subGroupSize = *r;

    for (int i = 0; i < 30; i++) {
      if (subGroupSize == 64)
        fast_hadamard_transform<T, 64>(params, q);
      else
        fast_hadamard_transform<T, 32>(params, q);
    }
    q.wait();

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      if (subGroupSize == 64)
        fast_hadamard_transform<T, 64>(params, q);
      else
        fast_hadamard_transform<T, 32>(params, q);
    }

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("batch size: %d | hidden dimension: %d | Average kernel execution time : %f (us)\n",
            batch_size, dim, (time * 1e-3f) / repeat);

    q.memcpy(h_out.data(), d_out, numel * sizeof(T)).wait();

    reference(h_x, r_out, batch_size, dim, scale);

    bool ok = true;
    for (int64_t i = 0; i < numel; ++i) {
      if (std::fabs((float)h_out[i] - (float)r_out[i]) > 1e-3f) {
        printf("Mismatch at index %ld %f %f\n", i, (float)h_out[i], (float)r_out[i]);
        ok = false;
        break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");

    sycl::free(d_x, q);
    sycl::free(d_out, q);
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <batch size> <repeat>\n", argv[0]);
    return 1;
  }
  const int batch_size = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  for (int dim = 8; dim <= 32768; dim = dim * 8) {
    hadamard_transform<float>(q, batch_size, dim, repeat);
    hadamard_transform<sycl::half>(q, batch_size, dim, repeat);
    hadamard_transform<sycl::ext::oneapi::bfloat16>(q, batch_size, dim, repeat);
  }
  if (!ok) return 1;
  return 0;
}
