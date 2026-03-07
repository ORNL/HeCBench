#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <hip/hip_runtime.h>
#include <chrono>
#include "reference.h"

template<int R>
__global__ void bilateralFilter(
    const float *__restrict__ in,
    float *__restrict__ out,
    int w, 
    int h, 
    float a_square,
    float variance_I,
    float variance_spatial)
{
  const int idx = blockIdx.x*blockDim.x + threadIdx.x;
  const int idy = blockIdx.y*blockDim.y + threadIdx.y;

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
      float weight = a_square * expf(spatial + range);

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

  // square of the height of the curve peak
  float a_square = 0.5f / (variance_I * (float)M_PI);

  int repeat = atoi(argv[5]);

  float *d_src, *d_dst;
  hipMalloc((void**)&d_dst, img_size * sizeof(float));
  hipMalloc((void**)&d_src, img_size * sizeof(float));

  float *h_src = (float*) malloc (img_size * sizeof(float));
  // host and device results
  float *h_dst = (float*) malloc (img_size * sizeof(float));
  float *r_dst = (float*) malloc (img_size * sizeof(float));

  srand(123);
  for (int i = 0; i < img_size; i++)
    h_src[i] = rand() % 256;

  hipMemcpy(d_src, h_src, img_size * sizeof(float), hipMemcpyHostToDevice); 

  dim3 threads (16, 16);
  dim3 blocks ((w+15)/16, (h+15)/16);

  hipDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    bilateralFilter<3><<<blocks, threads>>>(
        d_src, d_dst, w, h, a_square, variance_I, variance_spatial);

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (3x3) %f (ms)\n", (time * 1e-6f) / repeat);

  hipMemcpy(h_dst, d_dst, img_size * sizeof(float), hipMemcpyDeviceToHost); 

  // verify
  bool ok = true;
  reference<3>(h_src, r_dst, w, h, a_square, variance_I, variance_spatial);
  for (int i = 0; i < w*h; i++) {
    if (fabsf(r_dst[i] - h_dst[i]) > 1e-3) {
      ok = false;
      break;
    }
  }

  hipDeviceSynchronize();
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    bilateralFilter<6><<<blocks, threads>>>(
        d_src, d_dst, w, h, a_square, variance_I, variance_spatial);

  hipDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (6x6) %f (ms)\n", (time * 1e-6f) / repeat);

  hipMemcpy(h_dst, d_dst, img_size * sizeof(float), hipMemcpyDeviceToHost); 

  reference<6>(h_src, r_dst, w, h, a_square, variance_I, variance_spatial);
  for (int i = 0; i < w*h; i++) {
    if (fabsf(r_dst[i] - h_dst[i]) > 1e-3) {
      ok = false;
      break;
    }
  }

  hipDeviceSynchronize();
  start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++)
    bilateralFilter<9><<<blocks, threads>>>(
        d_src, d_dst, w, h, a_square, variance_I, variance_spatial);

  hipDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time (9x9) %f (ms)\n", (time * 1e-6f) / repeat);

  hipMemcpy(h_dst, d_dst, img_size * sizeof(float), hipMemcpyDeviceToHost); 

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
  hipFree(d_dst);
  hipFree(d_src);
  return 0;
}
