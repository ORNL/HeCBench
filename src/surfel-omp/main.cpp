#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <random>
#include <omp.h>
#include "reference.h"

// compute the xyz images using the inverse focal length invF
template<typename T>
void surfel_render(
    const T *__restrict s,
    int N,
    T f,
    int w,
    int h,
    T *__restrict d)
{
  #pragma omp target teams distribute parallel for collapse(2) thread_limit(256)
  for (int idy = 0; idy < h; idy++)
    for (int idx = 0; idx < w; idx++) {
      T ray[3];
      ray[0] = T(idx)-(w-1)*(T)0.5;
      ray[1] = T(idy)-(h-1)*(T)0.5;
      ray[2] = f;
      T pt[3];
      T n[3];
      T p[3];
      T dMin = 1e20;

      //#pragma omp parallel for reduction(min:dMin)
      for (int i=0; i<N; ++i) {
        p[0] = s[i*COL_DIM+COL_P_X];
        p[1] = s[i*COL_DIM+COL_P_Y];
        p[2] = s[i*COL_DIM+COL_P_Z];
        n[0] = s[i*COL_DIM+COL_N_X];
        n[1] = s[i*COL_DIM+COL_N_Y];
        n[2] = s[i*COL_DIM+COL_N_Z];
        T rSqMax = s[i*COL_DIM+COL_RSq];
        T pDotn = p[0]*n[0]+p[1]*n[1]+p[2]*n[2];
        T dsDotRay = ray[0]*n[0] + ray[1]*n[1] + ray[2]*n[2];
        T alpha = pDotn / dsDotRay;
        pt[0] = ray[0]*alpha - p[0];
        pt[1] = ray[1]*alpha - p[1];
        pt[2] = ray[2]*alpha - p[2];
        T t = ray[2]*alpha;
        T rSq = pt[0] * pt[0] + pt[1] * pt[1] + pt[2] * pt[2];
        if (rSq < rSqMax && dMin > t) {
          dMin = t; // ray hit the surfel 
        }
      }
      d[idy*w+idx] = dMin > (T)100 ? (T)0 : dMin;
    }
}

template <typename T>
void surfelRenderTest(int n, int w, int h, int repeat)
{
  const int src_size = n*7;
  const int dst_size = w*h;

  T *r_dst = (T*) malloc (dst_size * sizeof(T));
  T *h_dst = (T*) malloc (dst_size * sizeof(T));
  T *h_src = (T*) malloc (src_size * sizeof(T));

  std::mt19937 gen(19937);
  std::uniform_real_distribution<T> dis1(-5, 5);
  std::uniform_real_distribution<T> dis2(0.3, 5);
  std::uniform_real_distribution<T> dis3(-1, 1);
  std::uniform_real_distribution<T> dis4(4e-4, 2.5e-3);
  for (int i = 0; i < n; i++) {
      h_src[i*COL_DIM+COL_P_X] = dis1(gen);
      h_src[i*COL_DIM+COL_P_Y] = dis1(gen);
      h_src[i*COL_DIM+COL_P_Z] = dis2(gen);
      T nx = dis3(gen);
      T ny = dis3(gen);
      T nz = dis3(gen);
      T s = sqrt(nx*nx+ny*ny+nz*nz);
      h_src[i*COL_DIM+COL_N_X] = nx / s;
      h_src[i*COL_DIM+COL_N_Y] = ny / s;
      h_src[i*COL_DIM+COL_N_Z] = nz / s;
      h_src[i*COL_DIM+COL_RSq] = dis4(gen);
  }

  T inverseFocalLength[3] = {0.005, 0.02, 0.036};
  bool ok = true;

  #pragma omp target data map(to: h_src[0:src_size]) \
                          map(alloc: h_dst[0:dst_size])
  {
    for (int f = 0; f < 3; f++) {
      printf("\nf = %d\n", f);

      reference<T>(h_src, n, inverseFocalLength[f], w, h, r_dst);

      auto start = std::chrono::steady_clock::now();

      for (int i = 0; i < repeat; i++)
        surfel_render<T>(h_src, n, inverseFocalLength[f], w, h, h_dst);

      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);

      #pragma omp target update from (h_dst[0:dst_size])

      for (int i = 0; i < dst_size; i++) {
        if (fabs(h_dst[i] - r_dst[i]) > 1e-3) {
          printf("%f %f\n", h_dst[i] , r_dst[i]);
          ok = false;
          break;
        }
      }
      if (!ok) break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(r_dst);
  free(h_dst);
  free(h_src);
}

int main(int argc, char *argv[]) {
  if (argc != 5) {
    printf("Usage: %s <number of surfels> <output width> <output height> <repeat>\n", argv[0]);
    return 1;
  }
  int n = atoi(argv[1]);
  int w = atoi(argv[2]);
  int h = atoi(argv[3]);
  int repeat = atoi(argv[4]);

  printf("-------------------------------------\n");
  printf(" surfelRenderTest with type float32  \n");
  printf("-------------------------------------\n");
  surfelRenderTest<float>(n, w, h, repeat);

  if (!ok) return 1;
  return 0;
}
