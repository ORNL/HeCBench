#include <chrono>
#include <cmath>
#include <cstdio>
#include <new>
#include <string>
#include <omp.h>

// thread block size
#define BSIZE 256

template <class T>
class AvgPoolGrad {
  public:
    void compute(const T& x, const T& y, const T& dy, T scale, T* dx) {
      *dx += (scale * dy);
    }
};

template <class T>
class MaxPoolGrad {
  public:
    void compute(const T& x, const T& y, const T& dy, T scale, T* dx) {
      *dx += dy * static_cast<T>(x == y);
    }
};

#include "reference.h"
#include <cstdlib>

template <typename PoolProcess, typename T>
void KernelPool2DGrad(
    const int nthreads,
    const T*__restrict input_data,
    const T*__restrict output_data,
    const T*__restrict output_grad,
    const int channels,
    const int input_height,
    const int input_width,
    const int output_height,
    const int output_width,
    const int ksize_height,
    const int ksize_width,
    const int stride_height,
    const int stride_width,
    const int padding_height,
    const int padding_width,
    PoolProcess pool_process,
    bool exclusive,
    T*__restrict input_grad,
    bool channel_last = false)
{
  #pragma omp target teams distribute parallel for thread_limit(BSIZE)
  for (int index = 0; index < nthreads; index ++) {
    int w_offset, h_offset, offsetC, batch_idx;
    int tmp;
    if (!channel_last) { /* NCHW */
      w_offset = index % input_width + padding_width;
      tmp = index / input_width;
      h_offset = tmp % input_height + padding_height;
      tmp = tmp / input_height;
      offsetC = tmp % channels;
      batch_idx = tmp / channels;
    } else { /* NHWC */
      offsetC = index % channels;
      tmp = index / channels;
      w_offset = tmp % input_width + padding_width;
      tmp = tmp / input_width;
      h_offset = tmp % input_height + padding_height;
      batch_idx = tmp / input_height;
    }

    int phstart, phend;
    int pwstart, pwend;
    phstart = (h_offset < ksize_height) ? 0 : (h_offset - ksize_height) / stride_height + 1;
    pwstart = (w_offset < ksize_width) ? 0 : (w_offset - ksize_width) / stride_width + 1;
    phend = std::min(h_offset / stride_height + 1, output_height);
    pwend = std::min(w_offset / stride_width + 1, output_width);

    // initial gradient value
    T gradient = static_cast<T>(0.0);
    T input = input_data[index];

    int output_stride = batch_idx * output_height * output_width * channels;
    if (!channel_last)
      output_stride += offsetC * output_height * output_width;

    const T *__restrict output_data_t = output_data + output_stride;
    const T *__restrict output_grad_t = output_grad + output_stride;

    for (int ph = phstart; ph < phend; ++ph) {
      for (int pw = pwstart; pw < pwend; ++pw) {
        int pool_size;
        int hstart = ph * stride_height - padding_height;
        int wstart = pw * stride_width - padding_width;
        int hend = std::min(hstart + ksize_height, input_height);
        int wend = std::min(wstart + ksize_width, input_width);
        hstart = std::max(hstart, 0);
        wstart = std::max(wstart, 0);
        pool_size = exclusive ? (hend - hstart) * (wend - wstart)
          : ksize_height * ksize_width;

        int output_sub_idx = channel_last
          ? (ph * output_width + pw) * channels + offsetC
          : ph * output_width + pw;
        pool_process.compute(input, output_data_t[output_sub_idx],
            output_grad_t[output_sub_idx],
            static_cast<T>(1.f / pool_size), &gradient);
      }
    }
    input_grad[index] = gradient;
  }
}

int main(int argc, char* argv[])
{
  if (argc != 8) {
    printf("Usage: %s <batch> <input channels> <input height> ", argv[0]);
    printf("<input width> <output height> <output width> <repeat>\n");
    return 1;
  }
  // input
  const int batch_size = atoi(argv[1]);
  const int input_channels = atoi(argv[2]);
  const int input_height = atoi(argv[3]);
  const int input_width = atoi(argv[4]);

  // output
  const int output_height = atoi(argv[5]);
  const int output_width = atoi(argv[6]);

  // repeat
  const int repeat = atoi(argv[7]);

  const int input_numel = batch_size*input_channels*input_height*input_width;
  const int output_numel = batch_size*input_channels*output_height*output_width;

  // attr
  const int ksize_height = 11;
  const int ksize_width = 11;
  const int stride_height = 4;
  const int stride_width = 4;
  const int padding_height = 1;
  const int padding_width = 1;
  const bool exclusive = true;
  const std::string data_format = "NCHW";
  const bool channel_last = (data_format == "NHWC");

  // problem size
  int nthreads = batch_size * input_channels * input_height * input_width;

  // process
  AvgPoolGrad<float> pool_process;

  float * input = new float[input_numel];
  float * output = new float[output_numel];
  float * output_grad = new float[output_numel];
  float * input_grad = new float[input_numel];
  float * input_grad_ref = new float[input_numel];

  srand(123);
  for (int i = 0; i < input_numel; ++i) {
    input[i] = (float)rand() / (float)RAND_MAX;
    input_grad[i] = 0.f;  // reset kernel output
  }

  for (int i = 0; i < output_numel; ++i) {
    output[i] = (float)rand() / (float)RAND_MAX;
    output_grad[i] = input_width * input_height;
  }

  #pragma omp target data map(to: input[0:input_numel], \
                                  output[0:output_numel],\
                                  output_grad[0:output_numel]) \
                          map(from: input_grad[0:input_numel])
  {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      KernelPool2DGrad<AvgPoolGrad<float>, float>(
        nthreads, input, output, output_grad, input_channels,
        input_height, input_width, output_height, output_width, ksize_height,
        ksize_width, stride_height, stride_width, padding_height, padding_width,
        pool_process, exclusive, input_grad, channel_last);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);
  }

  // verify
  reference<AvgPoolGrad<float>, float>(
          nthreads, input, output, output_grad,
          input_channels, input_height, input_width, output_height, output_width, ksize_height,
          ksize_width, stride_height, stride_width, padding_height, padding_width,
          pool_process, exclusive, input_grad_ref, channel_last);

  bool ok = true;
  for (int i = 0; i < input_numel; ++i) {
    if (fabsf(input_grad[i] - input_grad_ref[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  delete[] input;
  delete[] output;
  delete[] input_grad;
  delete[] input_grad_ref;
  delete[] output_grad;
  if (!ok) return 1;
  return 0;
}
