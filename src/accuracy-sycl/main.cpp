#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <random>
#include <sycl/sycl.hpp>
#include "reference.h"

#define GPU_NUM_THREADS 256

template <typename T>
void BlockReduce(T &input, sycl::nd_item<1> &item) {
  input = sycl::reduce_over_group(item.get_group(), input, sycl::plus<>());
}

void accuracy_kernel(
    sycl::nd_item<1> &item,
    const int N,
    const int D,
    const int top_k,
    const float* __restrict Xdata,
    const int* __restrict labelData,
    int* accuracy)
{
  int count = 0;

  for (int row = item.get_group(0); row < N; row += item.get_group_range(0)) {
    const int label = labelData[row];
    const float label_pred = Xdata[row * D + label];
    int ngt = 0;
    for (int col = item.get_local_id(0); col < D; col += item.get_local_range(0)) {
      const float pred = Xdata[row * D + col];
      if (pred > label_pred || (pred == label_pred && col <= label)) {
        ++ngt;
      }
    }
    BlockReduce(ngt, item);
    if (ngt <= top_k) {
      ++count;
    }
    item.barrier(sycl::access::fence_space::local_space);
  }
  if (item.get_local_id(0) == 0) {
    auto ao = sycl::atomic_ref<int,
                                sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space> (accuracy[0]);
    ao.fetch_add(count);
  }
}

void accuracy_kernel2(
    sycl::nd_item<1> &item,
    const int N,
    const int D,
    const int top_k,
    const float* __restrict Xdata,
    const int* __restrict labelData,
    int* accuracy,
    float &s_label_pred,
    int &s_label)
{
  int count = 0;

  for (int row = item.get_group(0); row < N; row += item.get_group_range(0)) {
    if (item.get_local_id(0) == 0) {
      s_label = labelData[row];
      s_label_pred = Xdata[row * D + s_label];
    }
    item.barrier(sycl::access::fence_space::local_space);

    const int   label      = s_label;
    const float label_pred = s_label_pred;
    int ngt = 0;
    const float* row_ptr = Xdata + row * D;
    int col = item.get_local_id(0);
    for (; col + 3 * item.get_local_range(0) < D; col += 4 * item.get_local_range(0)) {
      float p0 = row_ptr[col];
      float p1 = row_ptr[col + item.get_local_range(0)];
      float p2 = row_ptr[col + 2 * item.get_local_range(0)];
      float p3 = row_ptr[col + 3 * item.get_local_range(0)];

      ngt += (p0 > label_pred || (p0 == label_pred && col                               <= label));
      ngt += (p1 > label_pred || (p1 == label_pred && col +     item.get_local_range(0) <= label));
      ngt += (p2 > label_pred || (p2 == label_pred && col + 2 * item.get_local_range(0) <= label));
      ngt += (p3 > label_pred || (p3 == label_pred && col + 3 * item.get_local_range(0) <= label));
    }
    for (; col < D; col += item.get_local_range(0)) {
      float pred = row_ptr[col];
      ngt += (pred > label_pred || (pred == label_pred && col <= label));
    }

    BlockReduce(ngt, item);

    if (item.get_local_id(0) == 0 && ngt <= top_k) {
      ++count;
    }
    item.barrier(sycl::access::fence_space::local_space);
  }

  if (item.get_local_id(0) == 0 && count > 0) {
    auto ao = sycl::atomic_ref<int,
                                sycl::memory_order::relaxed,
                                sycl::memory_scope::device,
                                sycl::access::address_space::global_space> (accuracy[0]);
    ao.fetch_add(count);
  }
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: %s <number of rows> <number of columns> <top K> <repeat>\n", argv[0]);
    return 1;
  }
  const int nrows = atoi(argv[1]);
  const int ndims = atoi(argv[2]);
  const int top_k = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  const int data_size = nrows * ndims;

  const int label_size_bytes = nrows * sizeof(int);
  const size_t data_size_bytes = data_size * sizeof(float);

  int *label = (int*) malloc (label_size_bytes);

  srand(123);
  for (int i = 0; i < nrows; i++)
    label[i] = rand() % ndims;

  float *data = (float*) malloc (data_size_bytes);

  std::default_random_engine g (123);
  std::uniform_real_distribution<float> distr (0.f, 1.f);
  for (int i = 0; i < data_size; i++) {
    data[i] = distr(g);
  }

  int count_ref = reference(nrows, ndims, top_k, data, label);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  int *d_label = sycl::malloc_device<int>(nrows, q);
  q.memcpy(d_label, label, label_size_bytes);

  float *d_data = sycl::malloc_device<float>(data_size, q);
  q.memcpy(d_data, data, data_size_bytes);

  int *d_count = sycl::malloc_device<int>(1, q);

  q.wait();
  sycl::range<1> lws (GPU_NUM_THREADS);

  for (int ngrid = nrows / 4; ngrid <= nrows; ngrid += nrows / 4) {

    printf("Grid size is %d\n", ngrid);
    sycl::range<1> gws (ngrid * GPU_NUM_THREADS);

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      q.memset(d_count, 0, sizeof(int));
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class accuracy>(
          sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          accuracy_kernel(item, nrows, ndims, top_k, d_data, d_label, d_count);
        });
      });
    }

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of accuracy kernel: %f (us)\n", (time * 1e-3f) / repeat);

    int count;
    q.memcpy(&count, d_count, sizeof(int)).wait();
    printf("%s\n", (count == count_ref) ? "PASS" : "FAIL");

    // printf("Accuracy = %f\n", (float)count / nrows);
    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      q.memset(d_count, 0, sizeof(int));
      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<float, 0> s_label_pred_acc(cgh);
        sycl::local_accessor<int, 0> s_label_acc(cgh);
        cgh.parallel_for<class accuracy2>(
          sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          accuracy_kernel2(item, nrows, ndims, top_k, d_data, d_label, d_count,
                           s_label_pred_acc, s_label_acc);
        });
      });
    }

    q.wait();
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of accuracy kernel2: %f (us)\n", (time * 1e-3f) / repeat);

    q.memcpy(&count, d_count, sizeof(int)).wait();
    printf("%s\n", (count == count_ref) ? "PASS" : "FAIL");
  }

  sycl::free(d_label, q);
  sycl::free(d_data, q);
  sycl::free(d_count, q);

  free(label);
  free(data);

  return 0;
}
