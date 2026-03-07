/*
  Reference
  Chapter 16 in Programming massively parallel processors,
  A hands-on approach (D. Kirk and W. Hwu)
 */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <iostream>
#include <hip/hip_runtime.h>

#define TILE_WIDTH 16

#define II(n,c,h,w) ((n)*C*Hin*Win+(c)*Hin*Win+(h)*Win+w)
#define WI(n,c,h,w) ((n)*C*K*K+(c)*K*K+(h)*K+w)
#define OI(n,c,h,w) ((n)*M*Hout*Wout+(c)*Hout*Wout+(h)*Wout+w)

#ifdef MIOPEN_CONV
#include <miopen/miopen.h>
#define checkMIOpen(expression)                              \
  {                                                          \
    miopenStatus_t status = (expression);                    \
    if (status != miopenStatusSuccess) {                     \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << miopenGetErrorString(status) << std::endl;\
    }                                                        \
  }
#endif

template <typename T>
void verify (const T* Y, T* Y_ref, size_t Y_size)
{
  bool ok = true;
  for (size_t i = 0; i < Y_size; i++) {
    if (fabs(Y[i] - Y_ref[i]) > 1e-3f) {
      printf("%f (device) != %f (reference)\n", Y[i], Y_ref[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
}

template<typename T>
__global__
void conv3d_s1(const T * __restrict__ X,
               const T * __restrict__ W,
                     T * __restrict__ Y,
               const int C,
               const int M,
               const int K,
               const int Hin,
               const int Win,
               const int Hout,
               const int Wout,
               const int W_grid)
{
  int n = blockIdx.x;
  int m = blockIdx.y;
  int h = blockIdx.z / W_grid * TILE_WIDTH + threadIdx.y;
  int w = blockIdx.z % W_grid * TILE_WIDTH + threadIdx.x;
  if (h < Hout && w < Wout) {
    T s = 0;
    for (int c = 0; c < C; c++) {
      for (int p = 0; p < K; p++) {
        for (int q = 0; q < K; q++) {
          s += X[II(n, c, h+p, w+q)] * W[WI(m, c, p, q)];
        }
      }
    }
    Y[OI(n, m, h, w)] = s;
  }
}

template<typename T>
__global__
void conv3d_s2(const T * __restrict__ X,
               const T * __restrict__ W,
                     T * __restrict__ Y,
               const int C,
               const int M,
               const int K,
               const int Hin,
               const int Win,
               const int Hout,
               const int Wout,
               const int W_grid)
{
  int m = blockIdx.x;
  int h = blockIdx.y / W_grid * TILE_WIDTH + threadIdx.y;
  int w = blockIdx.y % W_grid * TILE_WIDTH + threadIdx.x;
  int n = blockIdx.z;
  if (h < Hout && w < Wout) {
    T s = 0;
    for (int c = 0; c < C; c++) {
      for (int p = 0; p < K; p++) {
        for (int q = 0; q < K; q++) {
          s += X[II(n, c, h+p, w+q)] * W[WI(m, c, p, q)];
        }
      }
    }
    Y[OI(n, m, h, w)] = s;
  }
}

template<typename T>
__global__
void conv3d_s3(const T * __restrict__ X,
               const T * __restrict__ W,
                     T * __restrict__ Y,
               const int C,
               const int M,
               const int K,
               const int Hin,
               const int Win,
               const int Hout,
               const int Wout,
               const int W_grid)
{
  int h = blockIdx.x / W_grid * TILE_WIDTH + threadIdx.y;
  int w = blockIdx.x % W_grid * TILE_WIDTH + threadIdx.x;
  int n = blockIdx.y;
  int m = blockIdx.z;
  if (h < Hout && w < Wout) {
    T s = 0;
    for (int c = 0; c < C; c++) {
      for (int p = 0; p < K; p++) {
        for (int q = 0; q < K; q++) {
          s += X[II(n, c, h+p, w+q)] * W[WI(m, c, p, q)];
        }
      }
    }
    Y[OI(n, m, h, w)] = s;
  }
}


// Hin = Hout-1+K; max(h+p) is Hin - 1 as max(h) = Hout-1 and max(p) = K-1
template <typename T>
void reference(const T * __restrict__ X,
               const T * __restrict__ W,
                     T * __restrict__ Y,
               const int N,
               const int M,
               const int C,
               const int K,
               const int Hin,
               const int Win,
               const int Hout,
               const int Wout)
{
  for(int n = 0; n < N; n++)
    for(int m = 0; m < M; m++)
      for(int h = 0; h < Hout; h++)
        for(int w = 0; w < Wout; w++) {
          Y[OI(n, m, h, w)] = 0;
          for(int c = 0; c < C; c++)
            for(int p = 0; p < K; p++)
              for(int q = 0; q < K; q++)
                Y[OI(n, m, h, w)] += X[II(n, c, h+p, w+q)] * W[WI(m, c, p, q)];
        }
}

template <typename T>
void conv3D(const int N, const int C, const int M, const int Win, const int Hin, const int K, const int repeat)
{
  const int Hout = Hin-K+1;
  const int Wout = Win-K+1;

  size_t X_size = N * C * Hin * Win;
  size_t W_size = M * C * K * K;
  size_t Y_size = N * M * Hout * Wout;
  size_t X_bytes = X_size * sizeof(T);
  size_t W_bytes = W_size * sizeof(T);
  size_t Y_bytes = Y_size * sizeof(T);

  T *X, *W, *Y, *Y_ref;
  X = (T *)malloc(X_bytes); // input
  W = (T *)malloc(W_bytes); // filter
  Y = (T *)malloc(Y_bytes); // output
  Y_ref = (T *)malloc(Y_bytes);

  srand(123);

  for (size_t i = 0; i < W_size; i++) W[i] = rand() % 31;
  for (size_t i = 0; i < X_size; i++) X[i] = rand() % 13;

  for (size_t i = 0; i < Y_size; i++) {
    Y[i] = -1;
    Y_ref[i] = -1;
  }

  reference(X, W, Y_ref, N, M, C, K, Hin, Win, Hout, Wout);

  T *dX, *dW, *dY;
  hipMalloc((void **)&dX, X_bytes);
  hipMalloc((void **)&dW, W_bytes);
  hipMalloc((void **)&dY, Y_bytes);

  hipMemcpy(dX, X, X_bytes, hipMemcpyHostToDevice);
  hipMemcpy(dW, W, W_bytes, hipMemcpyHostToDevice);
  hipMemcpy(dY, Y, Y_bytes, hipMemcpyHostToDevice);

  int W_grid = (Wout + TILE_WIDTH - 1) / TILE_WIDTH;
  int H_grid = (Hout + TILE_WIDTH - 1) / TILE_WIDTH;
  int Z = H_grid * W_grid;

  printf("input dimensions: C=%d Win=%d Hin=%d\n", C, Win, Hin);
  printf("output dimensions: M=%d Wout=%d Hout=%d\n", M, Wout, Hout);
  printf("3D grid dimensions: N=%d M=%d Z=%d\n", N, M, Z);

  // try grid organizations
  dim3 grids_s1 (N, M, Z);
  dim3 grids_s2 (M, Z, N);
  dim3 grids_s3 (Z, N, M);
  dim3 blocks (TILE_WIDTH, TILE_WIDTH, 1);

  hipDeviceSynchronize();

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    conv3d_s1 <<< grids_s1, blocks >>> (dX, dW, dY, C, M, K, Hin, Win, Hout, Wout, W_grid);
  }

  hipDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time of conv3d_s1 kernel: %f (us)\n",
         (time * 1e-3f) / repeat);
  hipMemcpy(Y, dY, Y_bytes, hipMemcpyDeviceToHost);
  verify(Y, Y_ref, Y_size);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    conv3d_s2 <<< grids_s2, blocks >>> (dX, dW, dY, C, M, K, Hin, Win, Hout, Wout, W_grid);
  }

  hipDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time of conv3d_s2 kernel: %f (us)\n",
         (time * 1e-3f) / repeat);
  hipMemcpy(Y, dY, Y_bytes, hipMemcpyDeviceToHost);
  verify(Y, Y_ref, Y_size);

  start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; i++) {
    conv3d_s3 <<< grids_s3, blocks >>> (dX, dW, dY, C, M, K, Hin, Win, Hout, Wout, W_grid);
  }

  hipDeviceSynchronize();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time of conv3d_s3 kernel: %f (us)\n",
         (time * 1e-3f) / repeat);
  hipMemcpy(Y, dY, Y_bytes, hipMemcpyDeviceToHost);
  verify(Y, Y_ref, Y_size);

#ifdef MIOPEN_CONV
  #include "conv3d_s4.cu"
  hipMemcpy(Y, dY, Y_bytes, hipMemcpyDeviceToHost);
  verify(Y, Y_ref, Y_size);
#endif

  free(X);
  free(W);
  free(Y);
  free(Y_ref);
  hipFree(dX);
  hipFree(dW);
  hipFree(dY);
}

int main(int argc, char* argv[]) {
  if (argc != 8) {
    printf("Usage: %s <batch size:N> <input channels:C> <output feature maps:M>", argv[0]);
    printf(" <input width:Win> <input height:Hin> <kernel size:K> <repeat>\n");
    return 1;
  }

  int N = atoi(argv[1]);
  int C = atoi(argv[2]);
  int M = atoi(argv[3]);
  int W = atoi(argv[4]);
  int H = atoi(argv[5]);
  int K = atoi(argv[6]);
  int repeat = atoi(argv[7]);

  printf("3D convolution (FP32)\n");
  printf("\n========== Warmup start ==========\n");
  conv3D<float>(N, C, M, W, H, K, 1000);
  printf("\n========== Warmup done ==========\n");
  conv3D<float>(N, C, M, W, H, K, repeat);

  if (!ok) return 1;
  return 0;
}
