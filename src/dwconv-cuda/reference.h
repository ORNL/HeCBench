#ifndef DWCONV_REFERENCE_H
#define DWCONV_REFERENCE_H

template <typename scalar_t, typename acc_t = scalar_t>
void reference(
    const scalar_t *input,
    scalar_t *output,
    const scalar_t *weight,
    const scalar_t *bias,
    const bool bias_enabled,
    const int batch_size,
    const int input_channels,
    const int input_height,
    const int input_width,
    const int output_channels,
    const int output_height,
    const int output_width,
    const int kernel_height,
    const int kernel_width,
    const int stride_height,
    const int stride_width,
    const int pad_height,
    const int pad_width,
    const int dilation_height,
    const int dilation_width)
{
  const int depthwise_multiplier = output_channels / input_channels;

  for (int n = 0; n < batch_size; ++n) {
    for (int c = 0; c < output_channels; ++c) {
      const int input_channel = c / depthwise_multiplier;
      for (int h = 0; h < output_height; ++h) {
        for (int w = 0; w < output_width; ++w) {
          acc_t value = bias_enabled
                            ? static_cast<acc_t>(bias[c])
                            : static_cast<acc_t>(0);

          for (int kh = 0; kh < kernel_height; ++kh) {
            const int input_h =
                -pad_height + h * stride_height + kh * dilation_height;
            for (int kw = 0; kw < kernel_width; ++kw) {
              const int input_w =
                  -pad_width + w * stride_width + kw * dilation_width;
              if (input_h >= 0 && input_h < input_height &&
                  input_w >= 0 && input_w < input_width) {
                const int input_index =
                    ((n * input_channels + input_channel) * input_height +
                     input_h) * input_width + input_w;
                const int weight_index =
                    (c * kernel_height + kh) * kernel_width + kw;
                value += static_cast<acc_t>(weight[weight_index]) *
                         static_cast<acc_t>(input[input_index]);
              }
            }
          }

          const int output_index =
              ((n * output_channels + c) * output_height + h) *
                  output_width + w;
          output[output_index] = static_cast<scalar_t>(value);
        }
      }
    }
  }
}

#endif
