void forward_pass(
    const float* __restrict__ in,
    float*       __restrict__ out,
    int N, float scale)
{
  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    float v = in[i] * scale;
    for (int iter = 0; iter < 256; iter++)
      v = v + ((iter + i) % 2 ? -1.f : 1.f);
    out[i] = sinf(v);
  }
}

// Simulates a backward pass gradient computation
void backward_pass(
    const float* __restrict__ fwd_out,
    const float* __restrict__ grad_in,
    float*       __restrict__ grad_out,
    int N)
{
  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    float g = grad_in[i];
    for (int iter = 0; iter < 256; iter++)
      g = g + ((iter + i) % 2 ? fwd_out[i] : -fwd_out[i]);
    grad_out[i] = sinf(g) / cosf(g);
  }
}

// Simulates data prefetch / augmentation (low priority, background)
void data_prefetch(
    float* __restrict__ buf,
    int N, int batch_id)
{
  #pragma omp parallel for
  for (int i = 0; i < N; i++) {
    float v = (float)(i + batch_id);
    for (int iter = 0; iter < 256; iter++)
      v = v + ((iter + i) % 2 ? -1.f : 1.f);
    buf[i] = v;
  }
}

// Simulates weight update
void sgd_update(
    float*       __restrict__ weights,
    const float* __restrict__ grads,
    int N, float lr)
{
  #pragma omp parallel for
  for (int i = 0; i < N; i++)
    weights[i] -= lr * grads[i];
}

