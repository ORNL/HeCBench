#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sycl/sycl.hpp>
#include <chrono>
#include <random>
#include "reference.h"

template <typename T, typename G>
void adam (
  sycl::nd_item<1> &item,
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
  const int i = item.get_global_id(0);
  const int totThreads = item.get_group_range(0) * item.get_local_range(0);

  for (size_t j = i; j < vector_size; j += totThreads) {
    for (int t = 1; t <= time_step; t++) {
      T scaled_grad = g[j]/grad_scale;
      m[j] = b1*m[j] + (1.f-b1)*scaled_grad;
      v[j] = b2*v[j] + (1.f-b2)*scaled_grad*scaled_grad;
      float m_corrected = m[j] / (1.f-sycl::pown(b1, t));
      float v_corrected = v[j] / (1.f-sycl::pown(b2, t));
      float denom;
      if (mode == ADAM_MODE_0)
        denom = sycl::sqrt(v_corrected + eps);
      else // Mode 1
        denom = sycl::sqrt(v_corrected) + eps;
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

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  std::mt19937 gen(19937);
  std::uniform_real_distribution<float> dist(0, 1);
  for (int i = 0; i < vector_size; i++) {
    m[i] = dist(gen);
    v[i] = dist(gen);
    g[i] = dist(gen);
    r[i] = p[i] = dist(gen);
  }

  float *d_m, *d_v, *d_g, *d_p;

  d_m = sycl::malloc_device<float>(vector_size, q);
  q.memcpy(d_m, m, size_bytes);

  d_v = sycl::malloc_device<float>(vector_size, q);
  q.memcpy(d_v, v, size_bytes);

  d_g = sycl::malloc_device<float>(vector_size, q);
  q.memcpy(d_g, g, size_bytes);

  d_p = sycl::malloc_device<float>(vector_size, q);
  q.memcpy(d_p, p, size_bytes);

  // Arbitrary constants
  const float step_size = 1e-3f;
  const float decay = 0.5f;
  const float beta1 = 0.9f;
  const float beta2 = 0.999f;
  const float eps = 1e-10f;
  const float grad_scale = 256.f;

  const int threadsPerBlock = 256;
  sycl::range<1> gws ((vector_size+threadsPerBlock-1) / threadsPerBlock * threadsPerBlock);
  sycl::range<1> lws (threadsPerBlock);

  adamMode_t mode = ADAM_MODE_0;

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class kernel>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        adam<float, float>(
          item,
          d_p, d_m, d_v, d_g,
          beta1, beta2,
          eps,
          grad_scale,
          step_size,
          time_step,
          vector_size,
          mode,
          decay);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (ms)\n", time * 1e-6f / repeat);

  q.memcpy(p, d_p, size_bytes).wait();

  sycl::free(d_p, q);
  sycl::free(d_m, q);
  sycl::free(d_v, q);
  sycl::free(d_g, q);

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
  if (!ok) exit(1);
  printf("Checksum: %lf %lf\n", cr / vector_size, cp / vector_size);

  free(p);
  free(m);
  free(v);
  free(g);
  free(r);
  return 0;
}
