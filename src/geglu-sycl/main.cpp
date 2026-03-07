#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <sycl/sycl.hpp>
#include "reference.h"

template<typename opmath_t>
opmath_t gelu(opmath_t x) {
    constexpr opmath_t kAlpha = M_SQRT1_2;
    return x * opmath_t(0.5) * (opmath_t(1) + sycl::erf(x * kAlpha));
}

template <typename scalar_t, typename opmath_t, int BLOCK_DIM_X, int DIM_LAST,
          int VEC_ELEMS, int FOR_LOOP>
void geglu_kernel(scalar_t *out, const scalar_t *x_and_gate,
                  const sycl::nd_item<1> &item) {
    static_assert(DIM_LAST % (BLOCK_DIM_X * VEC_ELEMS) == 0, "cannot decide CHUNKS_PER_ROW");
    constexpr int CHUNKS_PER_ROW = DIM_LAST / (BLOCK_DIM_X * VEC_ELEMS);
    struct alignas(sizeof(scalar_t) * VEC_ELEMS) U {
        scalar_t data[VEC_ELEMS];
    };
    U ux[FOR_LOOP];
    U ugate[FOR_LOOP];
    U uout[FOR_LOOP];
    for (int k = 0; k < FOR_LOOP; k++) {
        int idxN = (item.get_group(0) * FOR_LOOP + k) / CHUNKS_PER_ROW;
        int idxR = ((item.get_group(0) * FOR_LOOP + k) % CHUNKS_PER_ROW *
                     BLOCK_DIM_X + item.get_local_id(0)) * VEC_ELEMS;
        ux[k]    = *reinterpret_cast<U const *>(&x_and_gate[(idxN * 2 + 0) * (int64_t)DIM_LAST + idxR]);
        ugate[k] = *reinterpret_cast<U const *>(&x_and_gate[(idxN * 2 + 1) * (int64_t)DIM_LAST + idxR]);
    }
    for (int k = 0; k < FOR_LOOP; k++) {
        for (int i = 0; i < VEC_ELEMS; i++) {
            opmath_t gelu_out = gelu(static_cast<opmath_t>(ugate[k].data[i]));
            uout[k].data[i] = static_cast<scalar_t>(static_cast<opmath_t>(ux[k].data[i]) * gelu_out);
        }
    }
    for (int k = 0; k < FOR_LOOP; k++) {
        int idxN = (item.get_group(0) * FOR_LOOP + k) / CHUNKS_PER_ROW;
        int idxR = ((item.get_group(0) * FOR_LOOP + k) % CHUNKS_PER_ROW *
                     BLOCK_DIM_X + item.get_local_id(0)) * VEC_ELEMS;
        *reinterpret_cast<U *>(&out[idxN * (int64_t)DIM_LAST + idxR]) = uout[k];
    }
}

#define DISPATCH_DIM_LAST(VALUE, CONST_NAME, ...) [&] { \
    if (VALUE == 1280) { constexpr int CONST_NAME = 1280; return __VA_ARGS__(); } \
    if (VALUE == 2560) { constexpr int CONST_NAME = 2560; return __VA_ARGS__(); } \
    if (VALUE == 5120) { constexpr int CONST_NAME = 5120; return __VA_ARGS__(); } \
    throw std::invalid_argument("DISPATCH_DIM_LAST " + std::to_string(VALUE)); \
    }()

#define DISPATCH_FOR_LOOP(VALUE, CONST_NAME, ...) [&] { \
    if (VALUE == 1) { constexpr int CONST_NAME = 1; return __VA_ARGS__(); } \
    if (VALUE == 2) { constexpr int CONST_NAME = 2; return __VA_ARGS__(); } \
    throw std::invalid_argument("DISPATCH_FOR_LOOP " + std::to_string(VALUE)); \
    }()

template <typename scalar_t>
void geglu_gpu(sycl::queue &q, scalar_t *out, const scalar_t *x_and_gate, int64_t n, int dim_last)
{
    using opmath_t = float;
    constexpr int VEC_ELEMS = 8;
    constexpr int BLOCK_DIM_X = 160;
    int for_loop = 2;
    while (for_loop > 0 && n * dim_last % (BLOCK_DIM_X * VEC_ELEMS * for_loop) != 0) {
        for_loop /= 2;
    }
    if (for_loop == 0) {
        throw std::invalid_argument("cannot determine grid_dim");
    }
    sycl::range<1> gws (n * dim_last / (BLOCK_DIM_X * VEC_ELEMS * for_loop) * BLOCK_DIM_X);
    sycl::range<1> lws (BLOCK_DIM_X);
    DISPATCH_FOR_LOOP(for_loop, FOR_LOOP, [&] {
        DISPATCH_DIM_LAST(dim_last, DIM_LAST, [&] {
            q.parallel_for(sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
               geglu_kernel<scalar_t, opmath_t, BLOCK_DIM_X, DIM_LAST, VEC_ELEMS,
                   FOR_LOOP>(out, x_and_gate, item);
            });
        });
    });
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  std::default_random_engine generator(123);
  std::uniform_real_distribution<float> distribution(-6.f,6.f);

/*
    PyTorch reference
    dim_batch = x_and_gate.shape[:-1]
    dim_last = x_and_gate.shape[-1] // 2
    out = torch.empty(dim_batch + (dim_last,), dtype=x_and_gate.dtype, device=x_and_gate.device)
    geglu(out.data_ptr(), x_and_gate.data_ptr(), dim_batch.numel(), dim_last)
*/

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // verify and warmup
  float *x_and_gate, *d_x_and_gate, *output, *output_ref, *d_output;
  for (int batch = 1; batch <= 4; batch = batch * 4) {
    for (int shape = 128; shape <= 512; shape = shape * 2) {
      for (int dim_last = 1280; dim_last <= 1280; dim_last = dim_last * 2) {
          uint64_t nelems = (uint64_t)batch * shape * dim_last * 2;
          uint64_t nelems_bytes = nelems * sizeof(float);
          x_and_gate = (float*) malloc (nelems_bytes);
          output_ref = (float*) malloc (nelems_bytes / 2);
          output     = (float*) malloc (nelems_bytes / 2);

          for (uint64_t i = 0; i < nelems; i++) {
            x_and_gate[i] = distribution(generator);
          }
          geglu_reference(output_ref, x_and_gate, batch * shape, dim_last);

          d_x_and_gate = (float *)sycl::malloc_device(nelems_bytes, q);
          d_output = (float *)sycl::malloc_device(nelems_bytes / 2, q);
          q.memcpy(d_x_and_gate, x_and_gate, nelems_bytes);
          geglu_gpu(q, d_output, d_x_and_gate, batch * shape, dim_last);
          q.memcpy(output, d_output, nelems_bytes / 2).wait();

          bool ok = true;
          for (uint64_t i = 0; i < nelems/2; i++) {
            if (fabsf(output[i] - output_ref[i]) > 1e-3f) {
              ok = false;
              break;
            }
          }

          printf("%s\n", ok ? "PASS" : "FAIL");
          if (!ok) exit(1);
          free(x_and_gate);
          free(output);
          free(output_ref);
          sycl::free(d_x_and_gate, q);
          sycl::free(d_output, q);
      }
    }
  }

  for (int batch = 1; batch <= 16; batch = batch * 4) {
    for (int shape = 4096; shape <= 8192; shape = shape * 2) {
      for (int dim_last = 1280; dim_last <= 5120; dim_last = dim_last * 2) {
          uint64_t nelems = (uint64_t)batch * shape * dim_last * 2;
          uint64_t nelems_bytes = nelems * sizeof(float);
          x_and_gate = (float*) malloc (nelems_bytes);

          for (uint64_t i = 0; i < nelems; i++) {
            x_and_gate[i] = distribution(generator);
          }
          output = (float*) malloc (nelems_bytes / 2);
          d_x_and_gate = (float *)sycl::malloc_device(nelems_bytes, q);
          q.memcpy(d_x_and_gate, x_and_gate, nelems_bytes);
          d_output = (float *)sycl::malloc_device(nelems_bytes / 2, q);
          q.wait();
          auto start = std::chrono::steady_clock::now();
          for (int i = 0; i < repeat; i++) {
            geglu_gpu(q, d_output, d_x_and_gate, batch * shape, dim_last);
          }
          q.wait();
          auto end = std::chrono::steady_clock::now();
          auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
          printf("Batch size: %d, sequence length: %d, hidden dimension: %d\n", batch, shape, dim_last);
          printf("Average execution time of GeGLU kernel: %f (us)\n", (time * 1e-3f) / repeat);
          free(x_and_gate);
          free(output);
          sycl::free(d_x_and_gate, q);
          sycl::free(d_output, q);
      }
    }
  }

  return 0;
}
