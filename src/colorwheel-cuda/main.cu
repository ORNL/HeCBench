#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda.h>

// Color encoding of flow vectors
// adapted from the color circle idea described at
//   http://members.shaw.ca/quadibloc/other/colint.htm
//
// Daniel Scharstein, 4/2007

#define RY  15
#define YG  6
#define GC  4
#define CB  11
#define BM  13
#define MR  6
#define MAXCOLS  (RY + YG + GC + CB + BM + MR)
typedef unsigned char uchar;

__host__ __device__
void setcols(int cw[MAXCOLS][3], int r, int g, int b, int k)
{
  cw[k][0] = r;
  cw[k][1] = g;
  cw[k][2] = b;
}

__host__ __device__
void computeColor(float fx, float fy, uchar *pix)
{
  int cw[MAXCOLS][3];  // color wheel

  // relative lengths of color transitions:
  // these are chosen based on perceptual similarity
  // (e.g. one can distinguish more shades between red and yellow 
  //  than between yellow and green)
  int i;
  int k = 0;
  for (i = 0; i < RY; i++) setcols(cw, 255,     255*i/RY,   0,       k++);
  for (i = 0; i < YG; i++) setcols(cw, 255-255*i/YG, 255,     0,     k++);
  for (i = 0; i < GC; i++) setcols(cw, 0,       255,     255*i/GC,   k++);
  for (i = 0; i < CB; i++) setcols(cw, 0,       255-255*i/CB, 255,   k++);
  for (i = 0; i < BM; i++) setcols(cw, 255*i/BM,     0,     255,     k++);
  for (i = 0; i < MR; i++) setcols(cw, 255,     0,     255-255*i/MR, k++);

  float rad = sqrtf(fx * fx + fy * fy);
  float a = atan2f(-fy, -fx) / (float)M_PI;
  float fk = (a + 1.f) / 2.f * (MAXCOLS-1);
  int k0 = (int)fk;
  int k1 = (k0 + 1) % MAXCOLS;
  float f = fk - k0;
  for (int b = 0; b < 3; b++) {
    float col0 = cw[k0][b] / 255.f;
    float col1 = cw[k1][b] / 255.f;
    float col = (1.f - f) * col0 + f * col1;
    if (rad <= 1)
      col = 1.f - rad * (1.f - col); // increase saturation with radius
    else
      col *= .75f; // out of range
    pix[2 - b] = (int)(255.f * col);
  }
}

__global__
void color (uchar* pix, int size, int half_size, float range, float truerange)
{
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int x = blockDim.x * blockIdx.x + threadIdx.x;

  if (y < size && x < size) {
    float fx = (float)x / (float)half_size * range - range;
    float fy = (float)y / (float)half_size * range - range;
    if (x == half_size || y == half_size) return; // make black coordinate axes
    size_t idx = ((size_t)y * size + x) * 3;
    computeColor(fx/truerange, fy/truerange, pix+idx);
  }
}

int main(int argc, char **argv)
{
  if (argc != 4) {
    printf("Usage: %s <range> <size> <repeat>\n", argv[0]);
    exit(1);
  }
  const float truerange = atof(argv[1]);
  const int size = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  // make picture slightly bigger to show out-of-range coding
  float range = 1.04f * truerange;

  const int half_size = size/2;

  // create a test image showing the color encoding
  size_t imgSize = (size_t)size * size * 3;
  uchar* pix = (uchar*) malloc (imgSize);
  uchar* res = (uchar*) malloc (imgSize);

  memset(pix, 0, imgSize);

  for (int y = 0; y < size; y++) {
    for (int x = 0; x < size; x++) {
      float fx = (float)x / (float)half_size * range - range;
      float fy = (float)y / (float)half_size * range - range;
      if (x == half_size || y == half_size) continue; // make black coordinate axes
      size_t idx = ((size_t)y * size + x) * 3;
      computeColor(fx/truerange, fy/truerange, pix+idx);
    }
  }

  printf("Start execution on a device\n");
  uchar *d_pix;
  cudaMalloc((void**)&d_pix, imgSize);
  cudaMemset(d_pix, 0, imgSize);

  dim3 grids ((size+15)/16, (size+15)/16);
  dim3 blocks (16, 16);

  cudaDeviceSynchronize();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    color <<< grids, blocks >>> (d_pix, size, half_size, range, truerange);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time : %f (ms)\n", (time * 1e-6f) / repeat);

  cudaMemcpy(res, d_pix, imgSize, cudaMemcpyDeviceToHost);

  // verify with tolerance
  const int tolerance = 1; // Allow differences up to 1 in each color channel
  int fail = 0;
  int max_error = 0;
  for(size_t i = 0; i < imgSize; i++) {
    int e = std::abs(static_cast<int>(res[i]) - static_cast<int>(pix[i]));
    if (e > tolerance) {
      fail = 1;
      if (e > max_error) max_error = e;
    }
  }
  if (fail) {
    printf("Verification failed. Maximum error between host and device results: %d\n", max_error);
  } else {
    printf("PASS\n");
  }
  
  cudaFree(d_pix);
  free(pix);
  free(res);
  return 0;
}
