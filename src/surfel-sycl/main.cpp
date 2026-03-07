#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <random>
#include <sycl/sycl.hpp>
#include "reference.h"

template<typename T>
void surfel_render(
  const T *__restrict__ s,
  int N,
  T f,
  int w,
  int h,
  T *__restrict__ d,
  const sycl::nd_item<3> &item)
{
  const int x = item.get_global_id(2);
  const int y = item.get_global_id(1);

  if(x < w && y < h)
  {
    T ray[3];
    ray[0] = T(x)-(w-1)*(T)0.5;
    ray[1] = T(y)-(h-1)*(T)0.5;
    ray[2] = f;
    T pt[3];
    T n[3];
    T p[3];
    T dMin = 1e20;
    
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
    d[y*w+x] = dMin > (T)100 ? (T)0 : dMin;
  }
}

template<typename T, int TILE>
void surfel_render_tile(
   const T *__restrict__ s,
   int N,
   T f,
   int w,
   int h,
   T *__restrict__ d,
   const sycl::nd_item<3> &item,
   T *sh)
{
    const int x = item.get_global_id(2);
    const int y = item.get_global_id(1);

    if (x >= w || y >= h) return;

    // Camera ray
    T rayx = T(x) - (w - 1) * T(0.5);
    T rayy = T(y) - (h - 1) * T(0.5);
    T rayz = f;

    T dMin = 1e20;

    // Shared memory for surfels

    for (int base = 0; base < N; base += TILE) {

        int tid = item.get_local_id(1) * item.get_local_range(2) +
                  item.get_local_id(2);
        if (tid < TILE && base + tid < N) {
            #pragma unroll
            for (int k = 0; k < COL_DIM; ++k) {
                sh[tid * COL_DIM + k] = s[(base + tid) * COL_DIM + k];
            }
        }
        item.barrier(sycl::access::fence_space::local_space);

        int tileCount = sycl::min(TILE, N - base);

        #pragma unroll
        for (int i = 0; i < tileCount; ++i) {

            T px = sh[i * COL_DIM + COL_P_X];
            T py = sh[i * COL_DIM + COL_P_Y];
            T pz = sh[i * COL_DIM + COL_P_Z];

            T nx = sh[i * COL_DIM + COL_N_X];
            T ny = sh[i * COL_DIM + COL_N_Y];
            T nz = sh[i * COL_DIM + COL_N_Z];

            T rSqMax = sh[i * COL_DIM + COL_RSq];

            T dsDotRay = rayx * nx + rayy * ny + rayz * nz;
            T pDotn = px * nx + py * ny + pz * nz;
            T alpha = pDotn / dsDotRay;
            T t = rayz * alpha;

            T dx = rayx * alpha - px;
            T dy = rayy * alpha - py;
            T dz = rayz * alpha - pz;

            T rSq = dx*dx + dy*dy + dz*dz;
            if (rSq < rSqMax && t < dMin) {
                dMin = t;
            }
        }
    }

    d[y * w + x] = (dMin > T(100)) ? T(0) : dMin;
}

template <typename T>
void surfelRenderTest(sycl::queue &q, int n, int w, int h, int repeat)
{
  const int src_size = n*7;
  const int dst_size = w*h;

  T *d_src, *d_dst;
  d_dst = sycl::malloc_device<T>(dst_size, q);
  d_src = sycl::malloc_device<T>(src_size, q);

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

  q.memcpy(d_src, h_src, src_size * sizeof(T));

  sycl::range<3> lws (1, 16, 16);
  sycl::range<3> gws (1, (h+15)/16*16, (w+15)/16*16);

  bool ok = true;
  for (int f = 0; f < 3; f++) {
    printf("\nf = %d\n", f);

    reference<T>(h_src, n, inverseFocalLength[f], w, h, r_dst);

    q.wait();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      q.submit([&](sycl::handler &cgh) {
        auto inverseFocalLength_f = inverseFocalLength[f];
        cgh.parallel_for(sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> item) {
          surfel_render<T>(d_src, n, inverseFocalLength_f, w, h, d_dst, item);
        });
      });

    q.wait();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of surfel_render(base): %f (ms)\n", (time * 1e-6f) / repeat);

    q.memcpy(h_dst, d_dst, dst_size * sizeof(T)).wait();

    for (int i = 0; i < dst_size; i++) {
      if (fabs(h_dst[i] - r_dst[i]) > 1e-3) {
        printf("%f %f\n", h_dst[i] , r_dst[i]);
        ok = false;
        break;
      }
    }
    if (!ok) break;

    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      q.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<T, 1> sh_acc(sycl::range<1>(256 * COL_DIM), cgh);
        auto inverseFocalLength_f = inverseFocalLength[f];
        cgh.parallel_for(
          sycl::nd_range<3>(gws, lws), [=](sycl::nd_item<3> item) {
            surfel_render_tile<T, 256>(
                d_src, n, inverseFocalLength_f, w, h, d_dst, item,
                sh_acc.template get_multi_ptr<sycl::access::decorated::no>().get());
          });
      });

    q.wait();
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of surfel_render(tile): %f (ms)\n", (time * 1e-6f) / repeat);

    q.memcpy(h_dst, d_dst, dst_size * sizeof(T)).wait();
    for (int i = 0; i < dst_size; i++) {
      if (fabs(h_dst[i] - r_dst[i]) > 1e-3) {
        printf("%f %f\n", h_dst[i] , r_dst[i]);
        ok = false;
        break;
      }
    }
    if (!ok) break;
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  free(r_dst);
  free(h_dst);
  free(h_src);
  sycl::free(d_dst, q);
  sycl::free(d_src, q);
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

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  printf("-------------------------------------\n");
  printf(" surfelRenderTest with type float32  \n");
  printf("-------------------------------------\n");
  surfelRenderTest<float>(q, n, w, h, repeat);

  if (!ok) return 1;
  return 0;
}
