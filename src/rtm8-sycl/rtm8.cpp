#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sycl/sycl.hpp>

#define nx 680
#define ny 134
#define nz 450

#include "mysecond.c"

inline int indexTo1D(int x, int y, int z){
  return x + y*nx + z*nx*ny;
}

void rtm8_cpu(float* vsq, float* current_s, float* current_r, float* next_s, float* next_r, float* image, float* a, size_t N)
{
#ifdef _OPENMP
#pragma omp parallel for collapse(3)
#endif
  for (int z = 4; z < nz - 4; z++) {
    for (int y = 4; y < ny - 4; y++) {
      for (int x = 4; x < nx - 4; x++) {
        float div =
          a[0] * current_s[indexTo1D(x,y,z)] +
          a[1] * (current_s[indexTo1D(x+1,y,z)] + current_s[indexTo1D(x-1,y,z)] +
              current_s[indexTo1D(x,y+1,z)] + current_s[indexTo1D(x,y-1,z)] +
              current_s[indexTo1D(x,y,z+1)] + current_s[indexTo1D(x,y,z-1)]) +
          a[2] * (current_s[indexTo1D(x+2,y,z)] + current_s[indexTo1D(x-2,y,z)] +
              current_s[indexTo1D(x,y+2,z)] + current_s[indexTo1D(x,y-2,z)] +
              current_s[indexTo1D(x,y,z+2)] + current_s[indexTo1D(x,y,z-2)]) +
          a[3] * (current_s[indexTo1D(x+3,y,z)] + current_s[indexTo1D(x-3,y,z)] +
              current_s[indexTo1D(x,y+3,z)] + current_s[indexTo1D(x,y-3,z)] +
              current_s[indexTo1D(x,y,z+3)] + current_s[indexTo1D(x,y,z-3)]) +
          a[4] * (current_s[indexTo1D(x+4,y,z)] + current_s[indexTo1D(x-4,y,z)] +
              current_s[indexTo1D(x,y+4,z)] + current_s[indexTo1D(x,y-4,z)] +
              current_s[indexTo1D(x,y,z+4)] + current_s[indexTo1D(x,y,z-4)]);

        next_s[indexTo1D(x,y,z)] = 2*current_s[indexTo1D(x,y,z)] - next_s[indexTo1D(x,y,z)]
          + vsq[indexTo1D(x,y,z)]*div;
        div =
          a[0] * current_r[indexTo1D(x,y,z)] +
          a[1] * (current_r[indexTo1D(x+1,y,z)] + current_r[indexTo1D(x-1,y,z)] +
              current_r[indexTo1D(x,y+1,z)] + current_r[indexTo1D(x,y-1,z)] +
              current_r[indexTo1D(x,y,z+1)] + current_r[indexTo1D(x,y,z-1)]) +
          a[2] * (current_r[indexTo1D(x+2,y,z)] + current_r[indexTo1D(x-2,y,z)] +
              current_r[indexTo1D(x,y+2,z)] + current_r[indexTo1D(x,y-2,z)] +
              current_r[indexTo1D(x,y,z+2)] + current_r[indexTo1D(x,y,z-2)]) +
          a[3] * (current_r[indexTo1D(x+3,y,z)] + current_r[indexTo1D(x-3,y,z)] +
              current_r[indexTo1D(x,y+3,z)] + current_r[indexTo1D(x,y-3,z)] +
              current_r[indexTo1D(x,y,z+3)] + current_r[indexTo1D(x,y,z-3)]) +
          a[4] * (current_r[indexTo1D(x+4,y,z)] + current_r[indexTo1D(x-4,y,z)] +
              current_r[indexTo1D(x,y+4,z)] + current_r[indexTo1D(x,y-4,z)] +
              current_r[indexTo1D(x,y,z+4)] + current_r[indexTo1D(x,y,z-4)]);

        next_r[indexTo1D(x,y,z)] = 2 * current_r[indexTo1D(x,y,z)]
          - next_r[indexTo1D(x,y,z)] + vsq[indexTo1D(x,y,z)] * div;

        image[indexTo1D(x,y,z)] = next_s[indexTo1D(x,y,z)] * next_r[indexTo1D(x,y,z)];
      }
    }
  }
}


int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  const int ArraySize = nx * ny * nz;
  float* next_s = (float*)malloc(ArraySize * sizeof(float));
  float* current_s = (float*)malloc(ArraySize * sizeof(float));
  float* next_r = (float*)malloc(ArraySize * sizeof(float));
  float* current_r = (float*)malloc(ArraySize * sizeof(float));
  float* vsq = (float*)malloc(ArraySize * sizeof(float));
  float* image_gpu = (float*)malloc(ArraySize * sizeof(float));
  float* image_cpu = (float*)malloc(ArraySize * sizeof(float));

  float a[5];
  double pts, t0, t1, dt, flops, pt_rate, flop_rate, speedup, memory;

  memory = ArraySize*sizeof(float)*6;
  pts = (double)repeat*(nx-8)*(ny-8)*(nz-8);
  flops = 67*pts;
  printf("memory (MB) = %lf\n", memory/1e6);
  printf("pts (billions) = %lf\n", pts/1e9);
  printf("Tflops = %lf\n", flops/1e12);

  // Initialization of matrix
  a[0] = -1./560.;
  a[1] = 8./315;
  a[2] = -0.2;
  a[3] = 1.6;
  a[4] = -1435./504.;

  for (int z = 0; z < nz; z++) {
    for (int y = 0; y < ny; y++) {
      for (int x = 0; x < nx; x++) {
        vsq[indexTo1D(x,y,z)] = 1.0;
        next_s[indexTo1D(x,y,z)] = 0;
        current_s[indexTo1D(x,y,z)] = 1.0;
        next_r[indexTo1D(x,y,z)] = 0;
        current_r[indexTo1D(x,y,z)] = 1.0;
        image_gpu[indexTo1D(x,y,z)] = image_cpu[indexTo1D(x,y,z)] = 0.5;
      }
    }
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  //allocate and copy matrix to device
  float *vsq_d = sycl::malloc_device<float>(ArraySize, q);
  float *next_s_d = sycl::malloc_device<float>(ArraySize, q);
  float *next_r_d = sycl::malloc_device<float>(ArraySize, q);
  float *current_s_d = sycl::malloc_device<float>(ArraySize, q);
  float *current_r_d = sycl::malloc_device<float>(ArraySize, q);
  float *image_d = sycl::malloc_device<float>(ArraySize, q);
  float *a_d = sycl::malloc_device<float>(5, q);

  q.memcpy(vsq_d, vsq, ArraySize * sizeof(float));
  q.memcpy(next_s_d, next_s, ArraySize * sizeof(float));
  q.memcpy(current_s_d, current_s, ArraySize * sizeof(float));
  q.memcpy(next_r_d, next_r, ArraySize * sizeof(float));
  q.memcpy(current_r_d, current_r, ArraySize * sizeof(float));
  q.memcpy(image_d, image_gpu, ArraySize * sizeof(float));
  q.memcpy(a_d, a, 5 * sizeof(float));

  int groupSize = 16;
  int nx_pad = (nx + groupSize - 1) / groupSize * groupSize;
  int ny_pad = (ny + groupSize - 1) / groupSize * groupSize;
  int nz_pad = nz;

  sycl::range<3> gws (nz_pad, ny_pad, nx_pad);
  sycl::range<3> lws (1, groupSize, groupSize);

  q.wait();
  t0 = mysecond();

  for (int t = 0; t < repeat; t++) {
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<class kernel1>(
        sycl::nd_range<3>(gws, lws), [=] (sycl::nd_item<3> item) {
        int x = item.get_global_id(2);
        int y = item.get_global_id(1);
        int z = item.get_global_id(0);
        float div;
        if ((4 <= x && x < (nx - 4) ) && (4 <= y && y < (ny - 4)) && (4 <= z && z < (nz - 4))){
          div = a[0] * current_s_d[indexTo1D(x,y,z)] +
                a[1] * (current_s_d[indexTo1D(x+1,y,z)] + current_s_d[indexTo1D(x-1,y,z)] +
                  current_s_d[indexTo1D(x,y+1,z)] + current_s_d[indexTo1D(x,y-1,z)] +
                  current_s_d[indexTo1D(x,y,z+1)] + current_s_d[indexTo1D(x,y,z-1)]) +
                a[2] * (current_s_d[indexTo1D(x+2,y,z)] + current_s_d[indexTo1D(x-2,y,z)] +
                  current_s_d[indexTo1D(x,y+2,z)] + current_s_d[indexTo1D(x,y-2,z)] +
                  current_s_d[indexTo1D(x,y,z+2)] + current_s_d[indexTo1D(x,y,z-2)]) +
                a[3] * (current_s_d[indexTo1D(x+3,y,z)] + current_s_d[indexTo1D(x-3,y,z)] +
                  current_s_d[indexTo1D(x,y+3,z)] + current_s_d[indexTo1D(x,y-3,z)] +
                  current_s_d[indexTo1D(x,y,z+3)] + current_s_d[indexTo1D(x,y,z-3)]) +
                a[4] * (current_s_d[indexTo1D(x+4,y,z)] + current_s_d[indexTo1D(x-4,y,z)] +
                  current_s_d[indexTo1D(x,y+4,z)] + current_s_d[indexTo1D(x,y-4,z)] +
                  current_s_d[indexTo1D(x,y,z+4)] + current_s_d[indexTo1D(x,y,z-4)]);
          
          next_s_d[indexTo1D(x,y,z)] = 2*current_s_d[indexTo1D(x,y,z)] - next_s_d[indexTo1D(x,y,z)]
                                       + vsq_d[indexTo1D(x,y,z)]*div;
          div = a[0] * current_r_d[indexTo1D(x,y,z)] +
                a[1] * (current_r_d[indexTo1D(x+1,y,z)] + current_r_d[indexTo1D(x-1,y,z)] +
                    current_r_d[indexTo1D(x,y+1,z)] + current_r_d[indexTo1D(x,y-1,z)] +
                    current_r_d[indexTo1D(x,y,z+1)] + current_r_d[indexTo1D(x,y,z-1)]) +
                a[2] * (current_r_d[indexTo1D(x+2,y,z)] + current_r_d[indexTo1D(x-2,y,z)] +
                    current_r_d[indexTo1D(x,y+2,z)] + current_r_d[indexTo1D(x,y-2,z)] +
                    current_r_d[indexTo1D(x,y,z+2)] + current_r_d[indexTo1D(x,y,z-2)]) +
                a[3] * (current_r_d[indexTo1D(x+3,y,z)] + current_r_d[indexTo1D(x-3,y,z)] +
                    current_r_d[indexTo1D(x,y+3,z)] + current_r_d[indexTo1D(x,y-3,z)] +
                    current_r_d[indexTo1D(x,y,z+3)] + current_r_d[indexTo1D(x,y,z-3)]) +
                a[4] * (current_r_d[indexTo1D(x+4,y,z)] + current_r_d[indexTo1D(x-4,y,z)] +
                    current_r_d[indexTo1D(x,y+4,z)] + current_r_d[indexTo1D(x,y-4,z)] +
                    current_r_d[indexTo1D(x,y,z+4)] + current_r_d[indexTo1D(x,y,z-4)]);
          
          next_r_d[indexTo1D(x,y,z)] = 2 * current_r_d[indexTo1D(x,y,z)]
                                       - next_r_d[indexTo1D(x,y,z)] + vsq_d[indexTo1D(x,y,z)] * div;
          
          image_d[indexTo1D(x,y,z)] = next_s_d[indexTo1D(x,y,z)] * next_r_d[indexTo1D(x,y,z)];
        }
      });
    });
  }

  q.wait();
  t1 = mysecond();
  dt = t1 - t0;

  q.memcpy(image_gpu, image_d, ArraySize * sizeof(float)).wait();

  t0 = mysecond();
  for (int t = 0; t < repeat; t++) {
    rtm8_cpu(vsq, current_s, next_s, current_r, next_r, image_cpu, a, ArraySize);
  }
  t1 = mysecond();

  // verification
  bool ok = true;
  for (int i = 0; i < ArraySize; i++) {
    if (fabsf(image_cpu[i] - image_gpu[i]) > 0.1) {
      printf("@index %d host: %f device %f\n", i, image_cpu[i], image_gpu[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  pt_rate = pts/dt;
  flop_rate = flops/dt;
  speedup = (t1 - t0) / dt;
  printf("dt = %lf\n", dt);
  printf("pt_rate (millions/sec) = %lf\n", pt_rate/1e6);
  printf("flop_rate (Gflops) = %lf\n", flop_rate/1e9);
  printf("speedup over cpu = %lf\n", speedup);
  printf("average kernel execution time = %lf (s)\n", dt / repeat);

  //release arrays
  free(vsq);
  free(next_s);
  free(current_s);
  free(next_r);
  free(current_r);
  free(image_cpu);
  free(image_gpu);
  sycl::free(vsq_d, q);
  sycl::free(next_s_d, q);
  sycl::free(current_s_d, q);
  sycl::free(next_r_d, q);
  sycl::free(current_r_d, q);
  sycl::free(image_d, q);
  sycl::free(a_d, q);

  if (!ok) return 1;
  return 0;
}
