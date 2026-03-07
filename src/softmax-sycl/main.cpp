#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <sycl/sycl.hpp>

#define BLOCK_SIZE 256

// A C model derived from the OpenCL kernel
void softMax_cpu(const int numSlice, const int sliceSize, const float* src, float* dest) {
  for (int i = 0; i < numSlice; i++) {
    float max_ = src[i * sliceSize];
    for (int j = 1; j < sliceSize; j++) {
      max_ = (max_ < src[i * sliceSize + j]) ? src[i * sliceSize + j] : max_;
    }
    float sum = 0;
    for (int j = 0; j < sliceSize; j++) {
      float e = expf(src[i * sliceSize + j] - max_);
      sum += e;
      dest[i * sliceSize + j] = e;
    }
    for (int j = 0; j < sliceSize; j++) {
      dest[i * sliceSize + j] /= sum;
    }
  }
}

int main(int argc, char* argv[]) {
  if (argc != 5) {
    printf("Usage: %s <number of slices> <slice size> <implementations> <repeat>\n", argv[0]);
    printf("implementation 0: naive\n");
    printf("implementation 1: optimized\n");
    return 1;
  }

  int numSlice = atoi(argv[1]);
  int sliceSize = atoi(argv[2]);
  int kernel = atoi(argv[3]);
  int repeat = atoi(argv[4]);
  int numElem = numSlice * sliceSize;

  float* input = (float*) aligned_alloc(1024, sizeof(float) * numElem);
  float* output_gpu = (float*) aligned_alloc(1024, sizeof(float) * numElem);
  float* output_cpu = (float*) aligned_alloc(1024, sizeof(float) * numElem);

  srand(2);
  for (int i = 0; i < numSlice; i++)
    for (int j = 0; j < sliceSize; j++)
      input[i*sliceSize+j] = rand() % 13;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  auto sg_sizes = q.get_device().get_info<sycl::info::device::sub_group_sizes>();
  auto r = std::max_element(sg_sizes.begin(), sg_sizes.end());
  int warpSize = *r;

  float *d_input = sycl::malloc_device<float>(numElem, q);
  q.memcpy(d_input, input, sizeof(float) * numElem);

  float *d_output = sycl::malloc_device<float>(numElem, q);

  if (kernel == 1) {
    sycl::range<1> gws ((numSlice+BLOCK_SIZE/warpSize-1)/(BLOCK_SIZE/warpSize)*BLOCK_SIZE);
    sycl::range<1> lws (BLOCK_SIZE);

    q.wait();
    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
      q.submit([&](sycl::handler &h) {
        h.parallel_for<class sm2>(
          sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item)
          //[[sycl::reqd_sub_group_size(warpSize)]]
        {
          sycl::sub_group warp = item.get_sub_group();
          int i = item.get_group(0) * warp.get_group_linear_range() + warp.get_group_linear_id();
          if (i >= numSlice) return;
          float max_ = d_input[i * sliceSize];
          for (int j = warp.get_local_linear_id(); j < sliceSize; j += warp.get_max_local_range()[0]) {
            max_ = sycl::max(max_, d_input[i * sliceSize + j]);
          }
          max_ = sycl::reduce_over_group(warp, max_, sycl::maximum<float>{});
          float sum = 0;
          for (int j = warp.get_local_linear_id(); j < sliceSize; j += warp.get_max_local_range()[0]) {
            sum += sycl::exp(d_input[i * sliceSize + j] - max_);
          }
          sum = sycl::reduce_over_group(warp, sum, sycl::plus<float>{});
          for (int j = warp.get_local_linear_id(); j < sliceSize; j += warp.get_max_local_range()[0]) {
            d_output[i * sliceSize + j] = sycl::exp(d_input[i * sliceSize + j] - max_) / sum;
          }
        });
      });
    }
    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);
  }
  else {

    sycl::range<1> gws ((numSlice+BLOCK_SIZE-1)/BLOCK_SIZE*BLOCK_SIZE);
    sycl::range<1> lws (BLOCK_SIZE);

    q.wait();
    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
      q.submit([&](sycl::handler &h) {
        h.parallel_for<class sm>(
          sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
          int i = item.get_global_id(0);
          if (i >= numSlice) return;
          float max_ = d_input[i * sliceSize];
          for (int j = 1; j < sliceSize; j++) {
            max_ = sycl::max(max_, d_input[i * sliceSize + j]);
          }
          float sum = 0;
          for (int j = 0; j < sliceSize; j++) {
            sum += sycl::exp(d_input[i * sliceSize + j] - max_);
          }
          for (int j = 0; j < sliceSize; j++) {
            d_output[i * sliceSize + j] = sycl::exp(d_input[i * sliceSize + j] - max_) / sum;
          }
        });
      });
    }

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);
  }
  q.memcpy(output_gpu, d_output, sizeof(float) * numElem).wait();

  // verification
  bool ok = true;
  softMax_cpu(numSlice, sliceSize, input, output_cpu);
  for (int i = 0; i < numElem; i++) {
    if (fabsf(output_cpu[i] - output_gpu[i]) > 1e-3) {
      printf("@index %d host: %f device: %f\n", i, output_cpu[i], output_gpu[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  free(input);
  free(output_cpu);
  free(output_gpu);
  sycl::free(d_input, q);
  sycl::free(d_output, q);
  return 0;
}
