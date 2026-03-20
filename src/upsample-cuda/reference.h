void upsample_forward_reference(
    const float* x,
    float* out,
    uint64_t B, uint64_t C, uint64_t H, uint64_t W)
{
  auto H_out = H * 2;
  auto W_out = W * 2;
  auto img_size = H * W;
  auto img_out_size = H_out * W_out;

  #pragma omp parallel for collapse(4)
  for (uint64_t b = 0; b < B; b++) {
    for (uint64_t c = 0; c < C; c++) {
      for (uint64_t i = 0; i < H; i++) {
        for (uint64_t j = 0; j < W; j++) {

          // move pointers
          auto x_val = x[b * C * img_size + c * img_size + i * W + j];
          auto offset = b * C * img_out_size + c * img_out_size + 2 * i * W_out + 2 * j;

          for (int ii = 0; ii < 2; ii++) {
            for (int jj = 0; jj < 2; jj++) {
              out[offset + ii * W_out + jj] = x_val;
            }
          }
        }
      }
    }
  }
}

void upsample_backward_reference(
    const float* dout,
    float* dx,
    uint64_t B, uint64_t C, uint64_t H, uint64_t W)
{
  auto img_in_size = H * W;
  auto H_out = H*2;
  auto W_out = W*2;
  auto img_out_size = H_out * W_out;

  #pragma omp parallel for collapse(4)
  for (uint64_t b = 0; b < B; b++) {
    for (uint64_t c = 0; c < C; c++) {
      for (uint64_t i = 0; i < H; i++) {
        for (uint64_t j = 0; j < W; j++) {
          // move pointers
          auto dx_offset = b * C * img_in_size + c * img_in_size + i * W + j;
          auto dout_offset = b * C * img_out_size + c * img_out_size + 2 * i * W_out + 2 * j;
          auto dout_sum = 0.0f;
          for (int ii = 0; ii < 2; ii++) {
            for (int jj = 0; jj < 2; jj++) {
              dout_sum += dout[dout_offset + ii * W_out + jj];
            }
          }
          dx[dx_offset] = dout_sum;
        }
      }
    }
  }
}

