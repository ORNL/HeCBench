#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include "reference.h"

template<int R>
void bilateralFilter(
    const float *__restrict in,
    float *__restrict out,
    int w, 
    int h, 
    float a_square,
    float variance_I,
    float variance_spatial)
{
  #pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
  for (int idy = 0; idy < h; idy++)
    for (int idx = 0; idx < w; idx++) {

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
          float weight = a_square * expf(spatial + range);

          normalization += weight;
          res += (I_w * weight);
        }
      }
      out[id] = res/normalization;
    }
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

  // square of the height of the curve peak
  float a_square = 0.5f / (variance_I * (float)M_PI);

  int repeat = atoi(argv[5]);

  float *h_src = (float*) malloc (img_size * sizeof(float));
  // host and device results
  float *h_dst = (float*) malloc (img_size * sizeof(float));
  float *r_dst = (float*) malloc (img_size * sizeof(float));

  srand(123);
  for (int i = 0; i < img_size; i++)
    h_src[i] = rand() % 256;

  #pragma omp target data map(to: h_src[0:img_size]) \
                          map(alloc: h_dst[0:img_size])
  {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      bilateralFilter<3>(h_src, h_dst, w, h, a_square, variance_I, variance_spatial);

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (3x3) %f (ms)\n", (time * 1e-6f) / repeat);

    #pragma omp target update from (h_dst[0:img_size])

    // verify
    bool ok = true;
    reference<3>(h_src, r_dst, w, h, a_square, variance_I, variance_spatial);
    for (int i = 0; i < w*h; i++) {
      if (fabsf(r_dst[i] - h_dst[i]) > 1e-3) {
        ok = false;
        break;
      }
    }

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      bilateralFilter<6>(h_src, h_dst, w, h, a_square, variance_I, variance_spatial);

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (6x6) %f (ms)\n", (time * 1e-6f) / repeat);

    #pragma omp target update from (h_dst[0:img_size])

    reference<6>(h_src, r_dst, w, h, a_square, variance_I, variance_spatial);
    for (int i = 0; i < w*h; i++) {
      if (fabsf(r_dst[i] - h_dst[i]) > 1e-3) {
        ok = false;
        break;
      }
    }

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      bilateralFilter<9>(h_src, h_dst, w, h, a_square, variance_I, variance_spatial);

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (9x9) %f (ms)\n", (time * 1e-6f) / repeat);

    #pragma omp target update from (h_dst[0:img_size])

    reference<9>(h_src, r_dst, w, h, a_square, variance_I, variance_spatial);
    for (int i = 0; i < w*h; i++) {
      if (fabsf(r_dst[i] - h_dst[i]) > 1e-3) {
        ok = false;
        break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");
    if (!ok) exit(1);
  }

  free(h_dst);
  free(r_dst);
  free(h_src);
  return 0;
}
