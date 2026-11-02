#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <vector>
#include <omp.h>
#include "reference.h"

#define block_size 256

void swiglu_oai_kernel(
   const int rows,
   const int dim,
   const float alpha,
   const float limit,
   const float* Xdata,
   const float* Bdata,
         float* Ydata)
{
  const int64_t nelems = (int64_t)rows * dim;

  #pragma omp target teams distribute parallel for num_threads(block_size)
  for (int64_t index = 0; index < nelems; index++) {
    const int64_t i = index / dim;
    const int64_t d = index % dim;

    const int64_t xOffset = i * 2 * dim;

    float gate = Xdata[xOffset + 2 * d] + Bdata[2 * d];
    float linear = Xdata[xOffset + 2 * d + 1] + Bdata[2 * d + 1];

    gate = fminf(gate, limit);
    linear = fminf(fmaxf(linear, -limit), limit);

    const float s = 1.f / (1.f + expf(-alpha * gate));
    Ydata[i * dim + d] = gate * s * (linear + 1.f);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <number of rows> <last dimension> <repeat>\n", argv[0]);
    return 1;
  }

  const int rows = atoi(argv[1]);
  const int dim = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  if (rows <= 0 || dim <= 0) {
    fprintf(stderr, "Error: rows and dimension must be positive (got %d and %d)\n",
            rows, dim);
    return 1;
  }

  // The average kernel time is divided by the repeat count.
  if (repeat <= 0) {
    fprintf(stderr, "Error: repeat count must be positive (got %d)\n", repeat);
    return 1;
  }

  // Assert the rounding rules directly. The end-to-end comparison cannot catch
  // a tie-policy bug -- a tie sits exactly on a rounding boundary, where
  // values_match() grants slack by design -- so this is what guards it.
  if (!self_test_rounding()) return 1;

  printf("Shape of input tensor: ( %d %d )\n", rows, 2 * dim);

  const uint64_t x_nelems = (uint64_t)rows * 2 * dim;
  const uint64_t y_nelems = (uint64_t)rows * dim;
  const uint64_t b_nelems = (uint64_t)2 * dim;

  const uint64_t x_bytes = x_nelems * sizeof(float);
  const uint64_t y_bytes = y_nelems * sizeof(float);
  const uint64_t b_bytes = b_nelems * sizeof(float);

  float *X = (float*) malloc (x_bytes);
  float *B = (float*) malloc (b_bytes);
  float *Y = (float*) malloc (y_bytes);
  float *Y_ref = (float*) malloc (y_bytes);

  std::default_random_engine generator(123);
  // The range straddles the clamp limit so both clamped and unclamped
  // elements are exercised.
  std::uniform_real_distribution<float> distribution(-12.f, 12.f);

  for (uint64_t i = 0; i < x_nelems; i++) {
    X[i] = distribution(generator);
  }
  for (uint64_t i = 0; i < b_nelems; i++) {
    B[i] = distribution(generator);
  }

  ComputeSwigluOAI(rows, dim, kSwigluAlpha, kSwigluLimit, X, B, Y_ref);

  #pragma omp target data map(to: X[0:x_nelems], B[0:b_nelems]) \
                          map(from: Y[0:y_nelems])
  {
    // check correctness before benchmarking
    swiglu_oai_kernel(rows, dim, kSwigluAlpha, kSwigluLimit, X, B, Y);

    #pragma omp target update from (Y[0:y_nelems])

    bool ok = true;
    for (uint64_t i = 0; i < y_nelems; i++) {
      if (fabsf(Y[i] - Y_ref[i]) > 1e-3f) {
        ok = false;
        break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      swiglu_oai_kernel(rows, dim, kSwigluAlpha, kSwigluLimit, X, B, Y);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of SwiGLU (OAI) kernel: %f (us)\n",
           (time * 1e-3f) / repeat);
  }

  free(X);
  free(B);
  free(Y);
  free(Y_ref);

  return 0;
}
