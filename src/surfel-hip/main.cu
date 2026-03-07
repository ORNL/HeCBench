#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <random>
#include <hip/hip_runtime.h>
#include "reference.h"

template<typename T>
__global__ void surfel_render(
  const T *__restrict__ s,
  int N,
  T f,
  int w,
  int h,
  T *__restrict__ d)
{
  const int x = threadIdx.x + blockIdx.x*blockDim.x;
  const int y = threadIdx.y + blockIdx.y*blockDim.y;

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
__global__ void surfel_render_tile(
   const T *__restrict__ s,
   int N,
   T f,
   int w,
   int h,
   T *__restrict__ d)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= w || y >= h) return;

    // Camera ray
    T rayx = T(x) - (w - 1) * T(0.5);
    T rayy = T(y) - (h - 1) * T(0.5);
    T rayz = f;

    T dMin = 1e20;

    // Shared memory for surfels
    __shared__ T sh[TILE * COL_DIM];

    for (int base = 0; base < N; base += TILE) {

        int tid = threadIdx.y * blockDim.x + threadIdx.x;
        if (tid < TILE && base + tid < N) {
            #pragma unroll
            for (int k = 0; k < COL_DIM; ++k) {
                sh[tid * COL_DIM + k] = s[(base + tid) * COL_DIM + k];
            }
        }
        __syncthreads();

        int tileCount = min(TILE, N - base);

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
void surfelRenderTest(int n, int w, int h, int repeat)
{
  const int src_size = n*7;
  const int dst_size = w*h;

  T *d_src, *d_dst;
  hipMalloc((void**)&d_dst, dst_size * sizeof(T));
  hipMalloc((void**)&d_src, src_size * sizeof(T));

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

  hipMemcpy(d_src, h_src, src_size * sizeof(T), hipMemcpyHostToDevice); 

  dim3 threads(16, 16);
  dim3 blocks((w+15)/16, (h+15)/16);

  bool ok = true;
  for (int f = 0; f < 3; f++) {
    printf("\nf = %d\n", f);

    reference<T>(h_src, n, inverseFocalLength[f], w, h, r_dst);
    
    hipDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      surfel_render<T><<<blocks, threads>>>(d_src, n, inverseFocalLength[f], w, h, d_dst);

    hipDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of surfel_render(base): %f (ms)\n", (time * 1e-6f) / repeat);

    hipMemcpy(h_dst, d_dst, dst_size * sizeof(T), hipMemcpyDeviceToHost); 

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
      surfel_render_tile<T, 256><<<blocks, threads>>>(d_src, n, inverseFocalLength[f], w, h, d_dst);

    hipDeviceSynchronize();
    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of surfel_render(tile): %f (ms)\n", (time * 1e-6f) / repeat);

    hipMemcpy(h_dst, d_dst, dst_size * sizeof(T), hipMemcpyDeviceToHost); 
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
  if (!ok) exit(1);

  free(r_dst);
  free(h_dst);
  free(h_src);
  hipFree(d_dst);
  hipFree(d_src);
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

  return 0;
}
