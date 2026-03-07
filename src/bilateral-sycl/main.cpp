#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <sycl/sycl.hpp>
#include "reference.h"

template<int R>
void bilateralFilter(
    sycl::nd_item<2> &item,
    const float *__restrict in,
    float *__restrict out,
    int w, 
    int h, 
    float a_square,
    float variance_I,
    float variance_spatial)
{
  const int idx = item.get_global_id(1);
  const int idy = item.get_global_id(0);

  if(idx >= w || idy >= h) return;

  int id = idy*w + idx;
  float I = in[id];
  float res = 0;
  float normalization = 0;

  // window centered at the coordinate (idx, idy)
#ifdef LOOP_UNROLL
  #pragma unroll
#endif
  for(int i = -R; i <= R; i++) {
#ifdef LOOP_UNROLL
    #pragma unroll
#endif
    for(int j = -R; j <= R; j++) {

      int idk = idx+i;
      int idl = idy+j;

      // mirror edges
      if( idk < 0) idk = -idk;
      if( idl < 0) idl = -idl;
      if( idk > w - 1) idk = w - 1 - i;
      if( idl > h - 1) idl = h - 1 - j;

      int id_w = idl*w + idk;
      float I_w = in[id_w];

      // range kernel for smoothing differences in intensities
      float range = -(I-I_w) * (I-I_w) / (2.f * variance_I);

      // spatial (or domain) kernel for smoothing differences in coordinates
      float spatial = -((idk-idx)*(idk-idx) + (idl-idy)*(idl-idy)) /
                      (2.f * variance_spatial);

      // the weight is assigned using the spatial closeness (using the spatial kernel) 
      // and the intensity difference (using the range kernel)
      float weight = a_square * sycl::exp(spatial + range);

      normalization += weight;
      res += (I_w * weight);
    }
  }
  out[id] = res/normalization;
}

//
// reference https://en.wikipedia.org/wiki/Bilateral_filter
//
int main(int argc, char *argv[]) {

  if (argc != 6) {
    printf("Usage: %s <image width> <image height> <intensity> <spatial> <repeat>\n",
            argv[0]);
    return 1;
  }

  // image dimensions
  int w = atoi(argv[1]);
  int h = atoi(argv[2]);
  const int img_size = w*h;

   // As the range parameter increases, the bilateral filter gradually 
   // approaches Gaussian convolution more closely because the range 
   // Gaussian widens and flattens, which means that it becomes nearly
   // constant over the intensity interval of the image.
  float variance_I = atof(argv[3]);

   // As the spatial parameter increases, the larger features get smoothened.
  float variance_spatial = atof(argv[4]);

  int repeat = atoi(argv[5]);

  // square of the height of the curve peak
  float a_square = 0.5f / (variance_I * (float)M_PI);

  const size_t img_size_bytes = img_size * sizeof(float);

  float *h_src = (float*) malloc (img_size_bytes);
  // host and device results
  float *h_dst = (float*) malloc (img_size_bytes);
  float *r_dst = (float*) malloc (img_size_bytes);

  srand(123);
  for (int i = 0; i < img_size; i++)
    h_src[i] = rand() % 256;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_src = sycl::malloc_device<float>(img_size, q);
  q.memcpy(d_src, h_src, img_size_bytes);

  float *d_dst = sycl::malloc_device<float>(img_size, q);

  sycl::range<2> lws (16, 16);
  sycl::range<2> gws ((h+15)/16*16, (w+15)/16*16);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class radius3x3>(
        sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
        bilateralFilter<3>(item, d_src, d_dst,
                           w, h, a_square, variance_I, variance_spatial);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (3x3) %f (ms)\n", (time * 1e-6f) / repeat);

  q.memcpy(h_dst, d_dst, img_size_bytes).wait();

  // verify
  bool ok = true;
  reference<3>(h_src, r_dst, w, h, a_square, variance_I, variance_spatial);
  for (int i = 0; i < w*h; i++) {
    if (fabsf(r_dst[i] - h_dst[i]) > 1e-3) {
      ok = false;
      break;
    }
  }

  q.wait();
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class radius6x6>(
        sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
        bilateralFilter<6>(item, d_src, d_dst,
                           w, h, a_square, variance_I, variance_spatial);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (6x6) %f (ms)\n", (time * 1e-6f) / repeat);

  q.memcpy(h_dst, d_dst, img_size_bytes).wait();

  reference<6>(h_src, r_dst, w, h, a_square, variance_I, variance_spatial);
  for (int i = 0; i < w*h; i++) {
    if (fabsf(r_dst[i] - h_dst[i]) > 1e-3) {
      ok = false;
      break;
    }
  }

  q.wait();
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class radius9x9>(
        sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
        bilateralFilter<9>(item, d_src, d_dst,
                           w, h, a_square, variance_I, variance_spatial);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (9x9) %f (ms)\n", (time * 1e-6f) / repeat);

  q.memcpy(h_dst, d_dst, img_size_bytes).wait();

  reference<9>(h_src, r_dst, w, h, a_square, variance_I, variance_spatial);
  for (int i = 0; i < w*h; i++) {
    if (fabsf(r_dst[i] - h_dst[i]) > 1e-3) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  free(h_dst);
  free(r_dst);
  free(h_src);
  sycl::free(d_dst, q);
  sycl::free(d_src, q);
  return 0;
}
