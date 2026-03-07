#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <omp.h>
#include "reference.h"

template<typename opmath_t>
opmath_t gelu(opmath_t x) {
    constexpr opmath_t kAlpha = M_SQRT1_2;
    return x * opmath_t(0.5) * (opmath_t(1) + erf(x * kAlpha));
}

template <typename scalar_t>
void geglu_gpu(scalar_t *out, const scalar_t *x_and_gate, int64_t n, int dim_last)
{
  #pragma omp target teams distribute parallel for collapse(2) num_threads(160)
  for (int i = 0; i < n; i++) {
    for (int d = 0; d < dim_last; d++) {
      scalar_t ux = x_and_gate[(i*2 + 0) * dim_last + d];
      scalar_t ug = x_and_gate[(i*2 + 1) * dim_last + d];
      out[i * dim_last + d] = ux * gelu_reference(ug);
    }
  }
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

  // verify and warmup
  float *x_and_gate, *output, *output_ref;
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

          #pragma omp target data map(to: x_and_gate[0:nelems]) map(from: output[0:nelems/2])
          {
            geglu_gpu(output, x_and_gate, batch * shape, dim_last);
          }

          bool ok = true;
          for (uint64_t i = 0; i < nelems/2; i++) {
            if (fabsf(output[i] - output_ref[i]) > 1e-3f) {
              ok = false;
              break;
            }
          }

          printf("%s\n", ok ? "PASS" : "FAIL");
          free(x_and_gate);
          free(output);
          free(output_ref);
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
          #pragma omp target data map(to: x_and_gate[0:nelems]) map(alloc: output[0:nelems/2])
          {
             auto start = std::chrono::steady_clock::now();
             for (int i = 0; i < repeat; i++) {
               geglu_gpu(output, x_and_gate, batch * shape, dim_last);
             }
             auto end = std::chrono::steady_clock::now();
             auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
             printf("Batch size: %d, sequence length: %d, hidden dimension: %d\n", batch, shape, dim_last);
             printf("Average execution time of GeGLU kernel: %f (us)\n", (time * 1e-3f) / repeat);
          }
          free(x_and_gate);
          free(output);
      }
    }
  }

  if (!ok) return 1;
  return 0;
}
