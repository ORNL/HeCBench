#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <sycl/sycl.hpp>
#include "kernels.h"

/*
Reference
vip-token-centric-compression/src/t5/models/small_embedding/kernel.py

batch size = 4
vector dim = 5
>>> indexes = torch.randint(0, 3, size = (4, ))
tensor([1, 1, 2, 0])
>>> indexes = indexes[:, None].repeat(1, 5)
>>> indexes
tensor([[1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [0, 0, 0, 0, 0]])
outputs.scatter_add_(0, indexes, source)
*/
void scatter_add_reference (int batch_size, int vector_dim,
                            float *out, int *idx, float *src)
{
  for (int d = 0; d < vector_dim; d++) {
    for (int i = 0; i < batch_size; i++) {
      int index = idx[i];
      out[index * vector_dim + d] += src[i * vector_dim + d];
    }
  }
}

void index_accumulate(int batch_size, int output_size, int vector_dim, int repeat)
{
  int   *d_index;
  float *d_source;
  float *d_output;

  size_t source_size_bytes = batch_size * vector_dim * sizeof(float);
  size_t output_size_bytes = output_size * vector_dim * sizeof(float);
  size_t index_size_bytes = batch_size * sizeof(int);

  int* index = (int*) malloc (index_size_bytes);
  float* source = (float*) malloc (source_size_bytes);
  float* output = (float*) malloc (output_size_bytes);
  float* output_ref = (float*) malloc (output_size_bytes);

  srand(2);
  for (int i = 0; i < batch_size; i++) {
    index[i] = rand() % output_size;
  }

  for (int i = 0; i < batch_size * vector_dim; i++) {
    source[i] = -1.f; // or random values
  }

  memset(output_ref, 0, output_size_bytes);
  scatter_add_reference (batch_size, vector_dim, output_ref, index, source);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  d_source = (float *)sycl::malloc_device(source_size_bytes, q);
  q.memcpy(d_source, source, source_size_bytes);

  d_index = (int *)sycl::malloc_device(index_size_bytes, q);
  q.memcpy(d_index, index, index_size_bytes);

  d_output = (float *)sycl::malloc_device(output_size_bytes, q);

  auto sg_sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
  auto r = std::max_element(sg_sizes.begin(), sg_sizes.end());
  int warpSize = *r;
  int thread_x = warpSize;
  int thread_y = MAX_THREADS_PER_BLOCK / warpSize;
  int block_x = batch_size / WORK_SIZE + 1;
  sycl::range<3> lws (1, thread_y, thread_x);
  sycl::range<3> gws (1, thread_y, thread_x * block_x);
  int shared_mem = (output_size * vector_dim + MAX_THREADS_PER_BLOCK);

  auto k1Fn = [&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> sm(sycl::range<1>(shared_mem), cgh);
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> item) {
      scatterAdd_kernel(
              d_index, d_source, d_output, batch_size, output_size,
              vector_dim, item,
              sm.get_multi_ptr<sycl::access::decorated::no>().get());
    });
  };

  auto k2Fn = [&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> sm(sycl::range<1>(shared_mem), cgh);
    cgh.parallel_for(sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> item) {
      scatterAdd2_kernel(
              d_index, d_source, d_output, batch_size, output_size,
              vector_dim, item,
              sm.get_multi_ptr<sycl::access::decorated::no>().get());
    });
  };

  // verify and warmup
  for (int i = 0; i < 10; i++) {
    q.memset(d_output, 0, output_size_bytes);
    q.submit(k2Fn);
  }
  q.memcpy(output, d_output, output_size_bytes).wait();

  bool ok = true;
  for (int i = 0; i < output_size * vector_dim; i++) {
    if (fabsf(output[i] - output_ref[i]) > 1e-3f) {
      printf("output %d: %f %f\n", i, output[i], output_ref[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  int64_t time = 0;
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    //q.memset(d_output, 0, output_size_bytes);
    q.submit(k1Fn);
  }
  q.wait();

  auto end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of kernel1: %f (us)\n", (time * 1e-3f) / repeat);

  time = 0;
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    //q.memset(d_output, 0, output_size_bytes);
    q.submit(k2Fn);
  }
  q.wait();

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of kernel2: %f (us)\n", (time * 1e-3f) / repeat);

  sycl::free(d_source, q);
  sycl::free(d_output, q);
  sycl::free(d_index, q);
  free(source);
  free(index);
  free(output);
  free(output_ref);
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: %s <batch size> <output size> <vector dimension> <repeat>\n", argv[0]);
    return 1;
  }
  const int batch_size = atoi(argv[1]);
  const int output_size = atoi(argv[2]);
  const int vector_dim = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  printf("batch_size: %d\n", batch_size);
  printf("output_size (range of index values): %d\n", output_size);
  printf("vector_dimension: %d\n", vector_dim);

  index_accumulate (batch_size, output_size, vector_dim, repeat) ;
  return 0;
}
