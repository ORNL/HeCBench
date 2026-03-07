#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <chrono>
#include <random>
#include "reference.h"

template <typename T, typename G>
inline void adam (
        T* __restrict p,
        T* __restrict m,
        T* __restrict v,
  const G* __restrict g,
  const float b1,
  const float b2,
  const float eps,
  const float grad_scale,
  const float step_size,
  const int time_step,
  const size_t vector_size,
  adamMode_t mode,
  const float decay)
{
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (size_t j = 0; j < vector_size; j++) {
    for (int t = 1; t <= time_step; t++) {
      T scaled_grad = g[j]/grad_scale;
      m[j] = b1*m[j] + (1.f-b1)*scaled_grad;
      v[j] = b2*v[j] + (1.f-b2)*scaled_grad*scaled_grad;
      float m_corrected = m[j] / (1.f-powf(b1, t));
      float v_corrected = v[j] / (1.f-powf(b2, t));
      float denom;
      if (mode == ADAM_MODE_0)
        denom = sqrtf(v_corrected + eps);
      else // Mode 1
        denom = sqrtf(v_corrected) + eps;
      float update = (m_corrected/denom) + (decay*p[j]);
      p[j] -= (step_size*update);
    }
  }
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <vector size> <number of time steps> <repeat>\n", argv[0]);
    return 1;
  }

  const int vector_size = atoi(argv[1]);
  const int time_step = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  size_t size_bytes = vector_size * sizeof(float);

  float *m = (float*) malloc (size_bytes);
  float *v = (float*) malloc (size_bytes);
  float *g = (float*) malloc (size_bytes);
  float *p = (float*) malloc (size_bytes);
  float *r = (float*) malloc (size_bytes);

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dist(0, 1);
  for (int i = 0; i < vector_size; i++) {
    m[i] = dist(gen);
    v[i] = dist(gen);
    g[i] = dist(gen);
    r[i] = p[i] = dist(gen);
  }

  // Arbitrary constants
  const float step_size = 1e-3f;
  const float decay = 0.5f;
  const float beta1 = 0.9f;
  const float beta2 = 0.999f;
  const float eps = 1e-10f;
  const float grad_scale = 256.f;

  adamMode_t mode = ADAM_MODE_0;

  #pragma omp target data map (to: m[0:vector_size], v[0:vector_size], g[0:vector_size]) \
                          map (tofrom: p[0:vector_size])
  {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      adam<float, float>(
        p, m, v, g,
        beta1, beta2,
        eps,
        grad_scale,
        step_size,
        time_step,
        vector_size,
        mode,
        decay);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %f (ms)\n", time * 1e-6f / repeat);
  }

  // verify
  reference<float, float>(
    repeat,
    r, m, v, g,
    beta1, beta2,
    eps,
    grad_scale,
    step_size,
    time_step,
    vector_size,
    mode,
    decay);

  bool ok = true; 
  double cr = 0, cp = 0;
  for (int i = 0; i < vector_size; i++) {
    if (fabsf(r[i] - p[i]) > 1e-3f) {
      ok = false;
      break;
    }
    cr += r[i]; cp += p[i];
  }

  printf("%s\n", ok ? "PASS" : "FAIL");
  printf("Checksum: %lf %lf\n", cr / vector_size, cp / vector_size);

  free(p);
  free(m);
  free(v);
  free(g);
  free(r);
  if (!ok) return 1;
  return 0;
}
