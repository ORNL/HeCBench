#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <random>
#include <sycl/sycl.hpp>
#include "reference.h"

#define H2F(input) static_cast<accscalar_t>(input)
#define F2H(input) static_cast<scalar_t>(input)

#define DEVICE_LINEAR_GET(A, INDEX) A[INDEX]
#define DEVICE_BIAS_GET(A, INDEX) A[INDEX]

template<typename T>
inline T sigmoid(T in)  {
  T one = static_cast<T>(1.0);
  return one / (one + sycl::exp(-in));
}

template <typename scalar_t, typename accscalar_t, typename index_type>
void gru_cell_forward(
  sycl::nd_item<1> &item,
  scalar_t *__restrict Input,
  scalar_t *__restrict Hidden,
  scalar_t *__restrict Bias1,
  scalar_t *__restrict Bias2,
  scalar_t *__restrict _hx,   // h(t-1)
  scalar_t *__restrict _hy,   // h(t)
  scalar_t *__restrict storage,
  index_type hsz,
  index_type totalElements)
{
  index_type linearIndex = item.get_global_id(0);
  if (linearIndex < totalElements) {
    index_type offset = (linearIndex/hsz)*3*hsz+linearIndex%hsz;

    scalar_t ir = DEVICE_LINEAR_GET(Input,  offset+0*hsz);
    scalar_t ii = DEVICE_LINEAR_GET(Input,  offset+1*hsz);
    scalar_t in = DEVICE_LINEAR_GET(Input,  offset+2*hsz);
    scalar_t hr = DEVICE_LINEAR_GET(Hidden, offset+0*hsz);
    scalar_t hi = DEVICE_LINEAR_GET(Hidden, offset+1*hsz);
    scalar_t hn = DEVICE_LINEAR_GET(Hidden, offset+2*hsz);

    scalar_t hx = DEVICE_LINEAR_GET(_hx, linearIndex);
    scalar_t* hy = &DEVICE_LINEAR_GET(_hy, linearIndex);

    scalar_t b1r, b1i, b1n, b2r, b2i, b2n;

    b1r = DEVICE_BIAS_GET(Bias1, linearIndex%hsz+0*hsz);
    b1i = DEVICE_BIAS_GET(Bias1, linearIndex%hsz+1*hsz);
    b1n = DEVICE_BIAS_GET(Bias1, linearIndex%hsz+2*hsz);

    b2r = DEVICE_BIAS_GET(Bias2, linearIndex%hsz+0*hsz);
    b2i = DEVICE_BIAS_GET(Bias2, linearIndex%hsz+1*hsz);
    b2n = DEVICE_BIAS_GET(Bias2, linearIndex%hsz+2*hsz);

    offset = (linearIndex/hsz)*5*hsz+linearIndex%hsz;

    accscalar_t rg, ig, ng;

    // reset: ir = Wr * xt , hr = Ur * h(t-1)
    rg = sigmoid(H2F(ir) + H2F(hr) + H2F(b1r) + H2F(b2r));

    // update: ii = Wz * xt , hi = Uz * h(t-1)
    ig = sigmoid(H2F(ii) + H2F(hi) + H2F(b1i) + H2F(b2i));

    // in = Wh * xt, hn = Uh * h(t-1)
    ng = H2F(in) + H2F(b1n) + rg*( H2F(hn) + H2F(b2n) );
    ng = sycl::tanh(ng); // h'

    // z * h(t-1) + (1-z)*h', hx = h(t-1)
    *hy = F2H( ng + ig * ( H2F(hx)-ng ) );

    //save for backwards
    DEVICE_LINEAR_GET(storage, offset+0*hsz) = F2H(rg);
    DEVICE_LINEAR_GET(storage, offset+1*hsz) = F2H(ig);
    DEVICE_LINEAR_GET(storage, offset+2*hsz) = F2H(ng);
    DEVICE_LINEAR_GET(storage, offset+3*hsz) = hx;
    DEVICE_LINEAR_GET(storage, offset+4*hsz) = F2H(H2F(hn) + H2F(b2n));
  }
}

int main(int argc, char* argv[])
{
  if (argc != 4) {
    printf("Usage: %s <number of sequences> <hidden size> <repeat>\n", argv[0]);
    return 1;
  }
  const int vsz = atoi(argv[1]);
  const int hsz = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  int input_size = 3 * vsz * hsz;
  size_t input_size_bytes = input_size * sizeof(sycl::half);

  int hidden_size = 3 * vsz * hsz;
  size_t hidden_size_bytes = hidden_size * sizeof(sycl::half);

  int bias_size = 3 * hsz;
  size_t bias_size_bytes = bias_size * sizeof(sycl::half);

  int store_size = 5 * vsz * hsz;
  size_t store_size_bytes = store_size * sizeof(sycl::half);

  int state_size = vsz;
  size_t state_size_bytes = state_size * sizeof(sycl::half);

  sycl::half *h_input, *h_hidden, *h_input_bias, *h_hidden_bias;
  sycl::half *h_hy, *h_hx, *h_hy_ref, *h_store, *h_store_ref;
  h_input = (sycl::half*) malloc (input_size_bytes);
  h_hidden = (sycl::half*) malloc (hidden_size_bytes);
  h_input_bias = (sycl::half*) malloc (bias_size_bytes);
  h_hidden_bias = (sycl::half*) malloc (bias_size_bytes);
  h_hy = (sycl::half*) malloc (state_size_bytes);
  h_hx = (sycl::half*) malloc (state_size_bytes);
  h_store = (sycl::half*) malloc(store_size_bytes);

  h_hy_ref = (sycl::half*) malloc (state_size_bytes);
  h_store_ref = (sycl::half*) malloc(store_size_bytes);

  std::default_random_engine g (123);
  std::uniform_real_distribution<float> distr (-2.f, 2.f);

  for (int i = 0; i < input_size; i++) {
    h_input[i] = distr(g);
  }

  for (int i = 0; i < hidden_size; i++) {
    h_hidden[i] = distr(g);
  }

  for (int i = 0; i < bias_size; i++) {
    h_input_bias[i] = distr(g);
    h_hidden_bias[i] = distr(g);
  }

  for (int i = 0; i < state_size; i++) {
    h_hx[i] = distr(g);
  }

  sycl::half *d_input, *d_hidden, *d_input_bias, *d_hidden_bias;
  sycl::half *d_hx, *d_hy, *d_store;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  d_input = sycl::malloc_device<sycl::half>(input_size, q);
  q.memcpy(d_input, h_input, input_size_bytes);

  d_hidden = sycl::malloc_device<sycl::half>(hidden_size, q);
  q.memcpy(d_hidden, h_hidden, hidden_size_bytes);

  d_input_bias = sycl::malloc_device<sycl::half>(bias_size, q);
  q.memcpy(d_input_bias, h_input_bias, bias_size_bytes);

  d_hidden_bias = sycl::malloc_device<sycl::half>(bias_size, q);
  q.memcpy(d_hidden_bias, h_hidden_bias, bias_size_bytes);

  d_hx = sycl::malloc_device<sycl::half>(state_size, q);
  q.memcpy(d_hx, h_hx, state_size_bytes);

  d_hy = sycl::malloc_device<sycl::half>(state_size, q);

  d_store = sycl::malloc_device<sycl::half>(store_size, q);

  sycl::range<1> gws ((vsz + 255) / 256 * 256);
  sycl::range<1> lws (256);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class gru>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        gru_cell_forward<sycl::half, float, int> (
          item, d_input, d_hidden, d_input_bias, d_hidden_bias,
          d_hx, d_hy, d_store, hsz, vsz);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of gru_cell_forward: %f (us)\n",
          (time * 1e-3f) / repeat);

  q.memcpy(h_hy, d_hy, state_size_bytes);
  q.memcpy(h_store, d_store, store_size_bytes);
  q.wait();

  reference<sycl::half, float, int>(
    h_input, h_hidden, h_input_bias, h_hidden_bias,
      h_hx, h_hy_ref, h_store_ref, hsz, vsz);

  bool ok = true;
  for (int i = 0; i < state_size; i++) {
    if (fabsf((float)(h_hy[i]) - (float)(h_hy_ref[i])) > 1e-3f) {
      ok = false;
      break;
    }
  }
  for (int i = 0; i < store_size; i++) {
    if (fabsf((float)(h_store[i]) - (float)(h_store_ref[i])) > 1e-3f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  sycl::free(d_input, q);
  sycl::free(d_hidden, q);
  sycl::free(d_input_bias, q);
  sycl::free(d_hidden_bias, q);
  sycl::free(d_hx, q);
  sycl::free(d_hy, q);
  sycl::free(d_store, q);

  free(h_input);
  free(h_hidden);
  free(h_input_bias);
  free(h_hidden_bias);
  free(h_hx);
  free(h_hy);
  free(h_store);
  free(h_hy_ref);
  free(h_store_ref);

  return 0;
}
