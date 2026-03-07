#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <sycl/sycl.hpp>

// Reference
// https://pytorch.org/docs/stable/generated/torch.linalg.cross.html#torch.linalg.cross

template <typename T>
class cross1;

template <typename T>
class cross2;

template <typename T>
class cross3;

template <typename T, typename StrideType>
void cross_kernel(
    sycl::nd_item<1> &item,
    int numel,
          T* out,
    const T* x1,
    const T* x2,
    StrideType ostride,
    StrideType x1stride,
    StrideType x2stride)
{
  for (int i = item.get_global_id(0);
           i < numel; i += item.get_local_range(0) * item.get_group_range(0)) {

    auto* out_row = out + 3*i;
    const auto* x1_row = x1 + 3*i;
    const auto* x2_row = x2 + 3*i;

    const T val0 = (x1_row[1 * x1stride] * x2_row[2 * x2stride] -
                    x1_row[2 * x1stride] * x2_row[1 * x2stride]);

    const T val1 = (x1_row[2 * x1stride] * x2_row[0 * x2stride] -
                    x1_row[0 * x1stride] * x2_row[2 * x2stride]);

    const T val2 = (x1_row[0 * x1stride] * x2_row[1 * x2stride] -
                    x1_row[1 * x1stride] * x2_row[0 * x2stride]);

    out_row[0 * ostride] = val0;
    out_row[1 * ostride] = val1;
    out_row[2 * ostride] = val2;
  }
}

template <typename T, typename StrideType>
void cross2_kernel(
    sycl::nd_item<1> &item,
    int numel,
          T* out,
    const T* x1,
    const T* x2,
    StrideType ostride,
    StrideType x1stride,
    StrideType x2stride)
{
  for (int i = item.get_global_id(0);
           i < numel; i += item.get_local_range(0) * item.get_group_range(0)) {

    auto* out_row = out + 3*i;
    const auto* x1_row = x1 + 3*i;
    const auto* x2_row = x2 + 3*i;

    const T x1_c0 = x1_row[0 * x1stride];
    const T x1_c1 = x1_row[1 * x1stride];
    const T x1_c2 = x1_row[2 * x1stride];
    const T x2_c0 = x2_row[0 * x2stride];
    const T x2_c1 = x2_row[1 * x2stride];
    const T x2_c2 = x2_row[2 * x2stride];

    const T val0 = x1_c1 * x2_c2 - x1_c2 * x2_c1 ;

    const T val1 = x1_c2 * x2_c0 - x1_c0 * x2_c2 ;

    const T val2 = x1_c0 * x2_c1 - x1_c1 * x2_c0 ;

    out_row[0 * ostride] = val0;
    out_row[1 * ostride] = val1;
    out_row[2 * ostride] = val2;
  }
}

template <typename T>
void cross3_kernel(
    sycl::nd_item<1> &item,
    int numel,
          T* out,
    const T* x1,
    const T* x2)
{
  for (int i = item.get_global_id(0);
           i < numel; i += item.get_local_range(0) * item.get_group_range(0)) {

    auto* out_row = out + 3*i;
    const auto* x1_row = x1 + 3*i;
    const auto* x2_row = x2 + 3*i;

    const T x1_c0 = x1_row[0];
    const T x1_c1 = x1_row[1];
    const T x1_c2 = x1_row[2];
    const T x2_c0 = x2_row[0];
    const T x2_c1 = x2_row[1];
    const T x2_c2 = x2_row[2];

    const T val0 = x1_c1 * x2_c2 - x1_c2 * x2_c1 ;

    const T val1 = x1_c2 * x2_c0 - x1_c0 * x2_c2 ;

    const T val2 = x1_c0 * x2_c1 - x1_c1 * x2_c0 ;

    out_row[0] = val0;
    out_row[1] = val1;
    out_row[2] = val2;
  }
}


template <typename T>
void eval(const int nrows, const int repeat) {
  const int num_elems = nrows * 3;
  const int size_bytes = num_elems * sizeof(T); 

  T *a, *b, *o, *o2, *o3;
  a = (T*) malloc (size_bytes);
  b = (T*) malloc (size_bytes);
  o = (T*) malloc (size_bytes);
  o2 = (T*) malloc (size_bytes);
  o3 = (T*) malloc (size_bytes);

  std::default_random_engine g (123);
  std::uniform_real_distribution<T> distr (-2.f, 2.f);
  for (int i = 0; i < num_elems; i++) {
    a[i] = distr(g);
    b[i] = distr(g);
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  T *d_a, *d_b, *d_o;
  d_o = sycl::malloc_device<T>(num_elems, q);
  d_a = sycl::malloc_device<T>(num_elems, q);
  d_b = sycl::malloc_device<T>(num_elems, q);

  q.memcpy(d_a, a, size_bytes);
  q.memcpy(d_b, b, size_bytes);

  sycl::range<1> gws ((nrows + 255) / 256 * 256);
  sycl::range<1> lws (256);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class cross1<T>>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        cross_kernel(item, nrows, d_o, d_a, d_b, 1, 1, 1);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of cross1 kernel: %f (us)\n", (time * 1e-3f) / repeat);

  q.memcpy(o, d_o, size_bytes).wait();

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class cross2<T>>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        cross2_kernel(item, nrows, d_o, d_a, d_b, 1, 1, 1);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of cross2 kernel: %f (us)\n", (time * 1e-3f) / repeat);

  q.memcpy(o2, d_o, size_bytes).wait();

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class cross3<T>>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        cross3_kernel(item, nrows, d_o, d_a, d_b);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of cross3 kernel: %f (us)\n", (time * 1e-3f) / repeat);

  q.memcpy(o3, d_o, size_bytes).wait();

  bool ok = true;
  for (int i = 0; i < num_elems; i++) {
    if (fabs(o[i] - o2[i]) > 1e-3 || fabs(o[i] - o3[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  sycl::free(d_a, q);
  sycl::free(d_b, q);
  sycl::free(d_o, q);

  free(a);
  free(b);
  free(o);
  free(o2);
  free(o3);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of rows in a 2D tensor> <repeat>\n", argv[0]);
    return 1;
  }
  const int nrows = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  printf("=========== Data type is FP32 ==========\n");
  eval<float>(nrows, repeat);

  printf("=========== Data type is FP64 ==========\n");
  eval<double>(nrows, repeat);

  if (!ok) return 1;
  return 0;
}
