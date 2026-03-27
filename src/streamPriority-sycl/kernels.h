// Simulates a compute-heavy forward pass layer
void forward_pass_kernel(
    const float* __restrict__ in,
    float*       __restrict__ out,
    int N, float scale,
    sycl::nd_item<1> &item)
{
  int i = item.get_global_id(0);
  if (i < N) {
    float v = in[i] * scale;
    for (int iter = 0; iter < 256; iter++)
        v = v + ((iter + i) % 2 ? -1.f : 1.f);
    out[i] = sinf(v);
  }
}

// Simulates a backward pass gradient computation
void backward_pass_kernel(
    const float* __restrict__ fwd_out,
    const float* __restrict__ grad_in,
    float*       __restrict__ grad_out,
    int N,
    sycl::nd_item<1> &item)
{
  int i = item.get_global_id(0);
  if (i < N) {
    float g = grad_in[i];
    for (int iter = 0; iter < 256; iter++)
      g = g + ((iter + i) % 2 ? fwd_out[i] : -fwd_out[i]);
    grad_out[i] = sinf(g) / cosf(g);
  }
}

// Simulates data prefetch / augmentation (low priority, background)
void data_prefetch_kernel(
    float* __restrict__ buf,
    int N, int batch_id,
    sycl::nd_item<1> &item)
{
  int i = item.get_global_id(0);
  if (i < N) {
    float v = (float)(i + batch_id);
    for (int iter = 0; iter < 256; iter++)
      v = v + ((iter + i) % 2 ? -1.f : 1.f);
    buf[i] = v;
  }
}

// Simulates weight update
void sgd_update_kernel(
    float*       __restrict__ weights,
    const float* __restrict__ grads,
    int N, float lr,
    sycl::nd_item<1> &item)
{
  int i = item.get_global_id(0);
  if (i < N) weights[i] -= lr * grads[i];
}
