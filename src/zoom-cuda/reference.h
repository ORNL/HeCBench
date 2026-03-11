void zoom_in_reference(
    const float* input_tensor,
    float*       output_tensor,
    int input_h,    int input_w,
    int output_h,   int output_w,
    size_t pitch,
    int out_h_start, int out_h_end,
    int out_w_start, int out_w_end,
    int batch_size)
{
  float ratio_h = (float)input_h / output_h;
  float ratio_w = (float)input_w / output_w;

  for (int b = 0; b < batch_size; b++) {

    const float* in_img  = input_tensor  + b * pitch;
    float*       out_img = output_tensor + b * pitch;

    // Iterate over every output pixel
    for (int oh = 0; oh < output_h; oh++) {
      for (int ow = 0; ow < output_w; ow++) {

        // Skip pixels outside the active output window
        if (oh < out_h_start || oh >= out_h_end ||
            ow < out_w_start || ow >= out_w_end)
          continue;

        // Source region that maps into this output pixel
        int start_h = (int)floorf( oh      * ratio_h);
        int end_h   = (int)ceilf ((oh + 1) * ratio_h);
        int start_w = (int)floorf( ow      * ratio_w);
        int end_w   = (int)ceilf ((ow + 1) * ratio_w);

        int del_h = end_h - start_h;
        int del_w = end_w - start_w;

        // Area-average over the source patch
        float sum = 0.0f;
        for (int i = 0; i < del_h; i++) {
          int src_row = start_h + i;
          if (src_row >= input_h) continue; // clamp

          for (int j = 0; j < del_w; j++) {
            int src_col = start_w + j;
            if (src_col >= input_w) continue; // clamp

            sum += in_img[src_row * input_w + src_col];
          }
        }

        // Write output
        out_img[(oh - out_h_start) * input_w + (ow - out_w_start)]
          = sum / (del_h * del_w);
      }
    }
  }
}

void zoom_out_reference(
    const float* input_tensor,
    float*       output_tensor,
    int input_h,    int input_w,
    int output_h,   int output_w,
    size_t pitch,
    int out_h_start, int out_h_end,
    int out_w_start, int out_w_end,
    int batch_size)
{
  float ratio_h = (float)input_h / output_h;
  float ratio_w = (float)input_w / output_w;

  for (int b = 0; b < batch_size; b++) {

    const float* in_img  = input_tensor  + b * pitch;
    float*       out_img = output_tensor + b * pitch;

    for (int oh = 0; oh < output_h; oh++) {
      for (int ow = 0; ow < output_w; ow++) {

        // Source region that maps into output pixel (oh, ow)
        int start_h = (int)floorf( oh      * ratio_h);
        int end_h   = (int)ceilf ((oh + 1) * ratio_h);
        int start_w = (int)floorf( ow      * ratio_w);
        int end_w   = (int)ceilf ((ow + 1) * ratio_w);

        int del_h = end_h - start_h;
        int del_w = end_w - start_w;

        // Area-average over the source patch
        float sum = 0.0f;
        for (int i = 0; i < del_h; i++) {
          int src_row = start_h + i;
          if (src_row >= input_h) continue;

          for (int j = 0; j < del_w; j++) {
            int src_col = start_w + j;
            if (src_col >= input_w) continue;

            sum += in_img[src_row * input_w + src_col];
          }
        }

        // Output addressing — note (oh + out_h_start), (ow + out_w_start)
        // This places the downscaled result at the correct canvas offset.
        out_img[(oh + out_h_start) * input_w + (ow + out_w_start)]
          = sum / (del_h * del_w);
      }
    }
  }
}

void zoom_out_edge_pad_reference(
    float* output_tensor,
    int height, int width,
    size_t pitch,
    int no_padding_h_start,
    int no_padding_w_start,
    int no_padding_h_end,
    int no_padding_w_end,
    int batch_size)
{
  for (int b = 0; b < batch_size; b++) {

    float* img = output_tensor + b * pitch;

    for (int oh = 0; oh < height; oh++) {
      for (int ow = 0; ow < width; ow++) {

        if (oh < no_padding_h_start
            && ow >= no_padding_w_start
            && ow <  no_padding_w_end) {
          // Top pad - replicate from first valid row
          img[oh * width + ow] =
            img[no_padding_h_start * width + ow];

        } else if (oh >= no_padding_h_end
            && ow >= no_padding_w_start
            && ow <  no_padding_w_end) {
          // Bottom pad - replicate from last valid row
          img[oh * width + ow] =
            img[(no_padding_h_end - 1) * width + ow];

        } else if (ow <  no_padding_w_start
            && oh >= no_padding_h_start
            && oh <  no_padding_h_end) {
          // Left pad - replicate from first valid column
          img[oh * width + ow] =
            img[oh * width + no_padding_w_start];

        } else if (ow >= no_padding_w_end
            && oh >= no_padding_h_start
            && oh <  no_padding_h_end) {
          // Right pad - replicate from last valid column
          img[oh * width + ow] =
            img[oh * width + (no_padding_w_end - 1)];

        } else if (oh <  no_padding_h_start
            && ow <  no_padding_w_start) {
          // Top-left corner
          img[oh * width + ow] =
            img[no_padding_h_start * width + no_padding_w_start];

        } else if (oh <  no_padding_h_start
            && ow >= no_padding_w_end) {
          // Top-right corner
          img[oh * width + ow] =
            img[no_padding_h_start * width + (no_padding_w_end - 1)];

        } else if (oh >= no_padding_h_end
            && ow <  no_padding_w_start) {
          // Bottom-left corner
          img[oh * width + ow] =
            img[(no_padding_h_end - 1) * width + no_padding_w_start];

        } else if (oh >= no_padding_h_end
            && ow >= no_padding_w_end) {
          // Bottom-right corner
          img[oh * width + ow] =
            img[(no_padding_h_end - 1) * width + (no_padding_w_end - 1)];
        }
        // Pixels inside the valid window are left untouched.
      }
    }
  }
}
