#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <cuda.h>
#include "kernels.h"
#include "reference.h"

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <vector size> <number of time steps>\n", argv[0]);
    return 1;
  }

  // assume each vector element contains two 4-bit quantized numbers
  const long vector_size = atol(argv[1]);
  const int time_step = atoi(argv[2]);

  int64_t size_bytes = vector_size * 2 * sizeof(float);

  float *g = (float*) malloc (size_bytes);
  float *p = (float*) malloc (size_bytes);
  float *p_ref = (float*) malloc (size_bytes);
  float *m_qscale = (float*) malloc (size_bytes);
  float *v_qscale = (float*) malloc (size_bytes);
  int8_t *m = (int8_t*) malloc (vector_size);
  int8_t *v = (int8_t*) malloc (vector_size);
  float *r = (float*) malloc (size_bytes);

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dist(0, 1);
  for (int64_t i = 0; i < vector_size * 2; i++) {
    m_qscale[i] = dist(gen);
    v_qscale[i] = dist(gen);
    g[i] = dist(gen);
    r[i] = p[i] = p_ref[i] = dist(gen);
  }

  for (int64_t i = 0; i < vector_size; i++) {
    m[i] = 256 * dist(gen);
    v[i] = 256 * dist(gen);
  }

  float *d_g, *d_p, *d_m_qscale, *d_v_qscale;
  int8_t *d_m, *d_v;

  cudaMalloc((void**)&d_m_qscale, size_bytes);
  cudaMemcpy(d_m_qscale, m_qscale, size_bytes, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_v_qscale, size_bytes);
  cudaMemcpy(d_v_qscale, v_qscale, size_bytes, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_m, vector_size);
  cudaMemcpy(d_m, m, vector_size, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_v, vector_size);
  cudaMemcpy(d_v, v, vector_size, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_g, size_bytes);
  cudaMemcpy(d_g, g, size_bytes, cudaMemcpyHostToDevice);

  cudaMalloc((void**)&d_p, size_bytes);
  cudaMemcpy(d_p, p, size_bytes, cudaMemcpyHostToDevice);

  const int threadsPerBlock = 64; // fixed at 64
  const int blocksPerGrid = (vector_size+threadsPerBlock-1) / threadsPerBlock;
  const dim3 grids (blocksPerGrid);
  const dim3 blocks (threadsPerBlock);

  // default constants
  const float lr = 1e-3f;
  const float weight_decay = 1e-2f;
  const float beta1 = 0.9f;
  const float beta2 = 0.999f;
  const float eps = 1e-8f;
  const float resid_beta1 = 1.0f - beta1;
  const float resid_beta2 = 1.0f - beta2;
  const float weight_decay_update = 1.0f - lr * weight_decay;

  for (int step = 1; step <= time_step; step++) {

    const float correction1 = 1.0f - powf(beta1, step);
    const float correction2_sqrt = sqrtf(1.0f - powf(beta2, step));
    const float step_size = lr / correction1;

    fused_4bit_kernel<float, threadsPerBlock><<<grids, blocks>>>(
              d_p,
              d_g,
              d_m_qscale,
              d_v_qscale,
              d_m,
              d_v,
              beta1,
              beta2,
              lr,
              weight_decay,
              eps,
              step,
              vector_size,
              correction1,
              correction2_sqrt,
              step_size,
              weight_decay_update,
              resid_beta1,
              resid_beta2);

    reference<float, threadsPerBlock>(
              blocksPerGrid,
              p_ref,
              g,
              m_qscale,
              v_qscale,
              m,
              v,
              beta1,
              beta2,
              lr,
              weight_decay,
              eps,
              step,
              vector_size,
              correction1,
              correction2_sqrt,
              step_size,
              weight_decay_update,
              resid_beta1,
              resid_beta2);
  }

  cudaMemcpy(p, d_p, size_bytes, cudaMemcpyDeviceToHost); 
  float absmax_error = 0;
  for (int64_t i = 0; i < vector_size * 2; i++) {
    absmax_error = fmaxf(absmax_error, fabsf(p[i] - p_ref[i]));
  }
  printf("Absolute maximum error: %f\n", absmax_error);
  printf("%s\n", absmax_error > 1e-3f ? "FAIL" : "PASS");

  auto start = std::chrono::steady_clock::now();

  for (int step = 1; step <= time_step; step++) {

    const float correction1 = 1.0f - powf(beta1, step);
    const float correction2_sqrt = sqrtf(1.0f - powf(beta2, step));
    const float step_size = lr / correction1;

    fused_4bit_kernel<float, threadsPerBlock><<<grids, blocks>>>(
              d_p,
              d_g,
              d_m_qscale,
              d_v_qscale,
              d_m,
              d_v,
              beta1,
              beta2,
              lr,
              weight_decay,
              eps,
              step,
              vector_size,
              correction1,
              correction2_sqrt,
              step_size,
              weight_decay_update,
              resid_beta1,
              resid_beta2);
  }
  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (ms)\n", time * 1e-6f / time_step);

  cudaFree(d_p);
  cudaFree(d_m);
  cudaFree(d_v);
  cudaFree(d_m_qscale);
  cudaFree(d_v_qscale);
  cudaFree(d_g);

  free(p);
  free(p_ref);
  free(m_qscale);
  free(v_qscale);
  free(m);
  free(v);
  free(g);
  free(r);
  if (absmax_error > 1e-3f) return 1;
  return 0;
}
