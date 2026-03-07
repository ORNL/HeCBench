#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <sycl/sycl.hpp>
#include "reference.h"

// limits of integration
#define A 0
#define B 15

// row size is related to accuracy
#define ROW_SIZE 17
#define EPS      1e-7

inline double f(double x)
{
  return sycl::exp(x)*sycl::sin(x);
}

inline unsigned int getFirstSetBitPos(int n)
{
  return sycl::log2((float)(n&-n))+1;
}

void romberg(double a, double b, double *result, double *smem, sycl::nd_item<1> &item)
{
  int threadIdx_x = item.get_local_id(0);
  int blockIdx_x = item.get_group(0);
  int gridDim_x = item.get_group_range(0);
  int blockDim_x = item.get_local_range(0);

  double diff = (b-a)/gridDim_x, step;
  int k;
  int max_eval = (1<<(ROW_SIZE-1));
  b = a + (blockIdx_x+1)*diff;
  a += blockIdx_x*diff;

  step = (b-a)/max_eval;

  double local_col[ROW_SIZE];  // specific to the row size
  for(int i = 0; i < ROW_SIZE; i++) local_col[i] = 0.0;
  if(!threadIdx_x)
  {
    k = blockDim_x;
    local_col[0] = f(a) + f(b);
  }
  else
    k = threadIdx_x;

  for(; k < max_eval; k += blockDim_x)
    local_col[ROW_SIZE - getFirstSetBitPos(k)] += 2.0*f(a + step*k);

  for(int i = 0; i < ROW_SIZE; i++)
    smem[ROW_SIZE*threadIdx_x + i] = local_col[i];
  item.barrier(sycl::access::fence_space::local_space);

  if(threadIdx_x < ROW_SIZE)
  {
    double sum = 0.0;
    for(int i = threadIdx_x; i < blockDim_x*ROW_SIZE; i+=ROW_SIZE)
      sum += smem[i];
    smem[threadIdx_x] = sum;
  }

  if(!threadIdx_x)
  {
    double *table = local_col;
    table[0] = smem[0];

    for(int k = 1; k < ROW_SIZE; k++)
      table[k] = table[k-1] + smem[k];

    for(int k = 0; k < ROW_SIZE; k++)
      table[k]*= (b-a)/(1<<(k+1));

    for(int col = 0 ; col < ROW_SIZE-1 ; col++)
      for(int row = ROW_SIZE-1; row > col; row--)
        table[row] = table[row] + (table[row] - table[row-1])/((1<<(2*col+1))-1);

    result[blockIdx_x] = table[ROW_SIZE-1];
  }
}


int main( int argc, char** argv)
{
  if (argc != 4) {
    printf("Usage: %s <number of work-groups> ", argv[0]);
    printf("<work-group size> <repeat>\n");
    return 1;
  }
  const int nwg = atoi(argv[1]);
  const int wgs = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const int result_size_byte = nwg * sizeof(double);
  double *h_result = (double*) malloc (result_size_byte);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  double *d_result = sycl::malloc_device<double>(nwg, q);

  sycl::range<1> gws (nwg * wgs);
  sycl::range<1> lws (wgs);

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<double, 1> smem (sycl::range<1>(ROW_SIZE*wgs), cgh);
      cgh.parallel_for<class k>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        romberg(A, B, d_result, smem.get_multi_ptr<sycl::access::decorated::no>().get(), item);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", time * 1e-9f / repeat);

  // verify
  q.memcpy(h_result, d_result, sizeof(double) * nwg).wait();

  double sum = 0.0;
  for(int k = 0; k < nwg; k++) sum += h_result[k];

  double ref_sum = reference(f, A, B, ROW_SIZE, EPS);
  printf("%s\n", (fabs(sum - ref_sum) > EPS) ? "FAIL" : "PASS");

  free(h_result);
  sycl::free(d_result, q);
  if ((fabs(sum - ref_sum) > EPS)) return 1;
  return 0;
}
