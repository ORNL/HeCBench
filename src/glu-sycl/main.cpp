#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <vector>
#include <sycl/sycl.hpp>
#include "reference.h"

void glu_kernel(
   sycl::nd_item<1> &item,
   const int M,
   const int split_dim_size,
   const int N,
   const float* Xdata,
         float* Ydata)
{
  int index = item.get_global_id(0);
  if (index >= M * split_dim_size * N) return;

  const int xOffset = 2 * split_dim_size * N;
  const int yOffset = split_dim_size * N;
  const int i = index / split_dim_size / N;
  const int j = index / N % split_dim_size;
  const int k = index % N;
  const float x1 = Xdata[i * xOffset + j * N + k];
  const float x2 = Xdata[i * xOffset + (j + split_dim_size) * N + k];
  Ydata[i * yOffset + j * N + k] = x1 * (1.f / (1.f + sycl::exp(-x2)));
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <number of dimensions> <size of each dimension> <repeat>\n", argv[0]);
    return 1;
  }

  const int ndims = atoi(argv[1]);
  const int dim_size = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  std::vector<int> Xshape;

  for (int i = 0; i < ndims; i++) {
    Xshape.push_back(dim_size);
  }

  std::vector<int> Yshape;
  Yshape.insert(Yshape.end(), Xshape.begin(), Xshape.end());

  printf("Shape of input tensor: ( ");
  for (int i = 0; i < ndims; i++)
    printf("%d ", Xshape[i]);
  printf(")\n");

  uint64_t nelems = size_from_dim(0, Xshape);
  uint64_t nelems_bytes = nelems * sizeof(float);

  float *X = (float*) malloc (nelems_bytes);
  float *Y = (float*) malloc (nelems_bytes);
  float *Y_ref = (float*) malloc (nelems_bytes);

  std::default_random_engine generator(123);
  std::uniform_real_distribution<float> distribution(-6.f,6.f);

  for (uint64_t i = 0; i < nelems; i++) {
    X[i] = distribution(generator);
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_X = sycl::malloc_device<float>(nelems, q);
  q.memcpy(d_X, X, nelems_bytes);

  float *d_Y = sycl::malloc_device<float>(nelems, q);

  const int block_size = 256; 

  for (int input_dim = -1; input_dim < 3 * (ndims-1); input_dim++) {

    const int split_index = (input_dim == -1) ? (ndims - 1) : (input_dim % ndims);

    if (Yshape[split_index] % 2 != 0) {
      printf("Split dimension %d should be divided by two. Skip\n", Yshape[split_index]);
      continue;
    }
    const int split_dim_size = Yshape[split_index] / 2;
    const int m = size_to_dim(split_index, Xshape);
    const int n = size_from_dim(split_index + 1, Xshape);

    ComputeGlu(m, split_dim_size, n, X, Y_ref);

    sycl::range<1> gws ((m * split_dim_size * n + block_size - 1) / block_size * block_size);
    sycl::range<1> lws (block_size);

    q.wait();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class glu>(
          sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          glu_kernel(item, m, split_dim_size, n, d_X, d_Y);
        });
      });
    }

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of GLU kernel (split dimension = %d): %f (us)\n",
           split_index, (time * 1e-3f) / repeat);

    q.memcpy(Y, d_Y, nelems_bytes).wait();

    bool ok = true;
    for (uint64_t i = 0; i < nelems/2; i++) {
      if (fabsf(Y[i] - Y_ref[i]) > 1e-3f) {
        ok = false;
        break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
  }

  free(X);
  free(Y);
  free(Y_ref);
  sycl::free(d_X, q);
  sycl::free(d_Y, q);

  if (!ok) return 1;
  return 0;
}
