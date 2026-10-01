#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <math.h>
#include <hip/hip_runtime.h>
#include "reference.h"
#include "tensorAccessor.h"

template <typename T, size_t N, template <typename U> class PtrTraits = DefaultPtrTraits>
using PackedTensorAccessor32 = GenericPackedTensorAccessor<T, N, PtrTraits, int>;

template <int kSize, typename scalar_t, typename acc_t, typename index_t>
__global__ void conv_depthwise2d_forward_kernel(
    const PackedTensorAccessor32<scalar_t, 4, RestrictPtrTraits> input,
          PackedTensorAccessor32<scalar_t, 4, RestrictPtrTraits> output,
    const PackedTensorAccessor32<scalar_t, 4, RestrictPtrTraits> weight,
    const PackedTensorAccessor32<scalar_t, 1, RestrictPtrTraits> bias,
    bool biasEnabled,
    index_t totalElements,
    const int outputChannels,
    const int depthwiseMultiplier,
    const int inputWidth, const int inputHeight,
    const int outputWidth, const int outputHeight,
    const int kernelWidth, const int kernelHeight,
    const int strideWidth, const int strideHeight,
    const int padWidth, const int padHeight,
    const int dilationWidth, const int dilationHeight)
{
  const int KW_LIMIT = (kSize != 0) ? kSize : kernelWidth;
  const int KH_LIMIT = (kSize != 0) ? kSize : kernelHeight;

  index_t linearIndex = blockIdx.x * blockDim.x + threadIdx.x;
  if (linearIndex < totalElements) {
    //calculate n,c,h,w indices, replacing modulos by divide and multiply add,
    //result is same as would be in the code below
    //const int n = linearIndex / batchStride; //batchStride = outputChannels * outputHeight * outputWidth
    //const int c = (linearIndex / channelStride) % outputChannels; //channelStride = outputHeight * outputWidth
    //const int h = (linearIndex / outputWidth) % outputHeight;
    //const int w = linearIndex % outputWidth;

    int indtmp1 = linearIndex/outputWidth;
    const int w = linearIndex - indtmp1 * outputWidth;
    int indtmp2 = indtmp1/outputHeight;
    const int h = indtmp1 - indtmp2 * outputHeight;
    indtmp1 = indtmp2;
    indtmp2 = indtmp1/outputChannels;
    const int c = indtmp1 - indtmp2 * outputChannels;
    const int n = indtmp2;

    int inputChannel = c;
    int inputChannels = outputChannels;
    if (depthwiseMultiplier !=1) {
      inputChannel /= depthwiseMultiplier;
      inputChannels /= depthwiseMultiplier;
    }

    int weightOffset = c * kernelHeight * kernelWidth;

    acc_t value = biasEnabled ? static_cast<acc_t>(bias.data()[c]) : acc_t(0);
    const index_t offset0 = (n * inputChannels + inputChannel) * inputHeight * inputWidth;
    #pragma unroll
    for (int kH = 0; kH < KH_LIMIT; ++kH) {
      #pragma unroll
      for (int kW = 0; kW < KW_LIMIT; ++kW) {
        const int h_in = -padHeight + h * strideHeight + kH * dilationHeight;
        const int w_in = -padWidth + w * strideWidth + kW * dilationWidth;

        if ((h_in >= 0) && (h_in < inputHeight) && (w_in >= 0) && (w_in < inputWidth)) {
          const index_t offset = offset0 + h_in * inputWidth + w_in;
          value += (static_cast<acc_t>(weight.data()[weightOffset]) *
                    static_cast<acc_t>(input.data()[offset]));
        }
        ++weightOffset;
      }
    }
    output.data()[linearIndex] = static_cast<scalar_t>(value);
  }
}

template <typename scalar_t>
void dwconv2d_forward (const int m,
                       const int n,
                       const int input_channels,
                       const int H,
                       const int W,
                       const int kH,
                       const int kW,
                       const int repeat,
                       const int padH = 1,
                       const int padW = 1,
                       const int strideH = 1,
                       const int strideW = 1,
                       const int dilateH = 1,
                       const int dilateW = 1)
{
  // The number of output_channels is a multiple of input_channels
  const int output_channels = input_channels * m;

  // Weight Tensor is shape (output_channels, 1, kH, kW)
  int weight_size = output_channels * kH * kW;
  size_t weight_size_bytes = weight_size * sizeof(scalar_t);
  int weight_sizes[] = {output_channels, 1, kH, kW};
  int weight_stride_sizes[] = {kH * kW, kH * kW, kW, 1};

  // Input Tensor is shape (N, input_channels, H, W)
  int input_size = n * input_channels * H * W;
  size_t input_size_bytes = input_size * sizeof(scalar_t);
  int input_sizes[] = {n, input_channels, H, W};
  int input_stride_sizes[] = {input_channels*H*W, H*W, W, 1};

  // Bias has same number of channels as output
  int bias_size = output_channels;
  size_t bias_size_bytes = bias_size * sizeof(scalar_t);
  int bias_stride_sizes[] = {1};

  int pad_sizes[] = {padH, padW};
  int stride_sizes[] = {strideH, strideW};
  int dilation_sizes[] = {dilateH, dilateW};

  int height = input_sizes[2];  // H
  int width = input_sizes[3];   // W
  int outputChannels = weight_sizes[0]; // output_channels

  int output_sizes[4];
  int input_batch_size_dim = 0;
  int weight_output_channels_dim = 0;
  output_sizes[0] = input_sizes[input_batch_size_dim];
  output_sizes[1] = weight_sizes[weight_output_channels_dim];

  for (int d = 2; d < 4; d++) {
    auto dilation_ = dilation_sizes[d - 2];
    auto kernel = dilation_ * (weight_sizes[d] - 1) + 1;
    output_sizes[d] = (input_sizes[d] + (2 * pad_sizes[d - 2]) - kernel) / stride_sizes[d - 2] + 1;
  }

  int output_stride_sizes[] = {
    output_sizes[1] * output_sizes[2] * output_sizes[3],
    output_sizes[2] * output_sizes[3], output_sizes[3], 1};

  int outputHeight = output_sizes[2];
  int outputWidth = output_sizes[3];

  int inputChannels = input_sizes[1];
  int depthwiseMultiplier = outputChannels / inputChannels;

  int output_size = 1;
  for (int i = 0; i < 4; i++)
    output_size *= output_sizes[i];

  size_t output_size_bytes = output_size * sizeof(scalar_t);

  dim3 grid ((output_size + 255) / 256);
  dim3 block (256); 
  
  scalar_t *h_input = (scalar_t*) malloc (input_size_bytes);
  scalar_t *h_weight = (scalar_t*) malloc (weight_size_bytes);
  scalar_t *h_bias = (scalar_t*) malloc (bias_size_bytes);
  scalar_t *h_output = (scalar_t*) malloc (output_size_bytes);
  scalar_t *h_reference = (scalar_t*) malloc (output_size_bytes);

  srand(123);
  for (int i = 0; i < input_size; i++) {
    h_input[i] = rand() / (float)RAND_MAX;
  }
  for (int i = 0; i < weight_size; i++) {
    h_weight[i] = rand() / (float)RAND_MAX;
  }
  for (int i = 0; i < bias_size; i++) {
    h_bias[i] = rand() / (float)RAND_MAX;
  }

  scalar_t *d_input;
  hipMalloc((void**)&d_input, input_size_bytes);
  hipMemcpy(d_input, h_input, input_size_bytes, hipMemcpyHostToDevice);
  PackedTensorAccessor32<scalar_t, 4, RestrictPtrTraits> input_a (d_input, input_sizes, input_stride_sizes);

  scalar_t *d_weight;
  hipMalloc((void**)&d_weight, weight_size_bytes);
  hipMemcpy(d_weight, h_weight, weight_size_bytes, hipMemcpyHostToDevice);
  PackedTensorAccessor32<scalar_t, 4, RestrictPtrTraits> weight_a (d_weight, weight_sizes, weight_stride_sizes);

  scalar_t *d_output;
  hipMalloc((void**)&d_output, output_size_bytes);
  PackedTensorAccessor32<scalar_t, 4, RestrictPtrTraits> output_a (d_output, output_sizes, output_stride_sizes);

  bool has_bias = true;
  scalar_t *d_bias;
  hipMalloc((void**)&d_bias, bias_size_bytes);
  hipMemcpy(d_bias, h_bias, bias_size_bytes, hipMemcpyHostToDevice);
  PackedTensorAccessor32<scalar_t, 1, RestrictPtrTraits> bias_a (d_bias, &bias_size, bias_stride_sizes);

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    if (kW == 3 && kH == 3) {
      conv_depthwise2d_forward_kernel<3, scalar_t, scalar_t, int> <<<grid, block>>>(
        input_a, output_a, weight_a, bias_a, has_bias, output_size, outputChannels, depthwiseMultiplier,
        width, height, outputWidth, outputHeight,
        kW, kH, strideW, strideH, padW, padH, dilateW, dilateH);
    } else if (kW == 1 && kH == 1) {
      conv_depthwise2d_forward_kernel<1, scalar_t, scalar_t, int> <<<grid, block>>>(
        input_a, output_a, weight_a, bias_a, has_bias, output_size, outputChannels, depthwiseMultiplier,
        width, height, outputWidth, outputHeight,
        kW, kH, strideW, strideH, padW, padH, dilateW, dilateH);
    } else {
      conv_depthwise2d_forward_kernel<0, scalar_t, scalar_t, int> <<<grid, block>>>(
        input_a, output_a, weight_a, bias_a, has_bias, output_size, outputChannels, depthwiseMultiplier,
        width, height, outputWidth, outputHeight,
        kW, kH, strideW, strideH, padW, padH, dilateW, dilateH);
    }
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of dwconv2d_forward kernel: %f (ms)\n", 
          time * 1e-6f / repeat);

  hipMemcpy(h_output, d_output, output_size_bytes, hipMemcpyDeviceToHost);

  reference<scalar_t>(
      h_input, h_reference, h_weight, h_bias, has_bias,
      n, inputChannels, height, width, outputChannels, outputHeight, outputWidth,
      kH, kW, strideH, strideW, padH, padW, dilateH, dilateW);

  int errors = 0;
  for (int i = 0; i < output_size; i++) {
    const scalar_t tolerance = 1e-4f + 1e-4f * fabs(h_reference[i]);
    if (fabs(h_output[i] - h_reference[i]) > tolerance)
      errors++;
  }
  printf("%s\n", errors == 0 ? "PASS" : "FAIL");

  hipFree(d_input);
  hipFree(d_output);
  hipFree(d_weight);
  hipFree(d_bias);

  free(h_input);
  free(h_output);
  free(h_reference);
  free(h_weight);
  free(h_bias);
}

int main(int argc, char* argv[])
{
  if (argc != 6) {
    printf("Usage: %s <batch size> <number of input channels> ", argv[0]);
    printf("<input height> <input width> <repeat>\n");
    return 1;
  }

  const int n = atoi(argv[1]);
  const int c = atoi(argv[2]);
  const int h = atoi(argv[3]);
  const int w = atoi(argv[4]);
  const int repeat = atoi(argv[5]);

  // sweep over depth multipliers and kernel sizes
  for (int m = 1; m <= 4; m = m + 1) {
    for (int k = 1; k <= 5; k = k + 2) {
      printf("batch = %d, input channel = %d, height = %d, width = %d, ",
             n, c, h, w);
      printf("kernel size = %d, output channel = %d\n", k, m * c);
      dwconv2d_forward<float>(m, n, c, h, w, k, k, repeat);
    }
  }

  return 0;
}
