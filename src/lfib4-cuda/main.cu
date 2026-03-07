#include <stdio.h>
#include <cuda.h>
#include <chrono>
#include <cstdlib>

#define P1 55
#define P2 119
#define P3 179
#define P4 256

#define LWDR 32
#define LKNB 8

typedef unsigned int uint32_t;

void LFIB4(uint32_t n, uint32_t *x) {
  for (uint32_t k = P4; k < n; k++) {
    x[k] = x[k - P1] + x[k - P2] + x[k - P3] + x[k - P4];
  }
}

__global__ void firstColGPU(uint32_t *x, int s) {
  __shared__ uint32_t cx[2 * P4];

  uint32_t *px = &cx[P4];
  int myid = threadIdx.x;
  cx[myid] = x[myid];
  __syncthreads();

  for (int k = 1; k < s / P4; k++) {

    for (int i = 0; i < P4; i += LWDR) {
      if (myid < LWDR) {
        px[i + myid] = px[i + myid - P1] + px[i + myid - P2]
                     + px[i + myid - P3] + px[i + myid - P4];
      }
      __syncthreads();
    }

    x[k * P4 + myid] = cx[myid] = px[myid];
    __syncthreads();
  }
}

__global__ void colYGPU(uint32_t *y, int s) {
  __shared__ uint32_t cy[3 * P4];

  uint32_t *ay = &cy[P4 * 2];
  int myid = threadIdx.x;
  ay[myid] = y[2 * P4 + myid];
  __syncthreads();

  for (int k = 0; k < s / P4; k++) {

    cy[myid] = cy[myid + P4];
    cy[myid + P4] = ay[myid];
    __syncthreads();

    for (int i = 0; i < P4; i += LWDR) {
      if (myid < LWDR) {
        ay[i + myid] = ay[i + myid - P1] + ay[i + myid - P2]
          + ay[i + myid - P3] + ay[i + myid - P4];
      }
      __syncthreads();
    }
  }

  y[2 * P4 + myid] = cy[2 * P4 + myid];
  y[P4 + myid] = cy[P4 + myid];
  y[myid] = cy[myid];
}

__global__ void lastEntGPU(uint32_t *__restrict__ x, uint32_t *__restrict__ y, int s, int r) {

  __shared__ uint32_t a0[3 * P4];
  __shared__ uint32_t b0[2 * P4];
  __shared__ uint32_t c0[2 * P4];
  __shared__ uint32_t d0[2 * P4];

  uint32_t *a = a0 + P4;
  uint32_t *b = b0 + P4;
  uint32_t *c = c0 + P4;
  uint32_t *d = d0 + P4;

  int myid = threadIdx.x;

  a0[myid] = y[myid];
  __syncthreads();

  if (myid < P4)
    a0[myid + P4 * 2] = y[myid + P4 * 2];
  __syncthreads();

  d0[myid] = c0[myid] = b0[myid] = a[myid];
  __syncthreads();

  b[myid - P4] += a[-(P4 - P3) + myid];
  __syncthreads();

  c[myid - P4] += (a[-(P3 - P2) + myid] + a[-(P4 - P2) + myid]);
  __syncthreads();

  d[myid - P4] += (a[-(P2 - P1) + myid] + a[-(P3 - P1) + myid]
      + a[-(P4 - P1) + myid]);
  __syncthreads();

  a += P4;

  for (int i = 1; i < r; i++) {

    uint32_t *xc = &x[i * s];
    uint32_t tmp = 0;

    if (myid < P4) {

      for (int k = 0; k < P4 - P3; k++)
        tmp += xc[-P4 + k] * a[myid - k];

      for (int k = 0; k < P3 - P2; k++)
        tmp += xc[-P3 + k] * b[myid - k];

      for (int k = 0; k < P2 - P1; k++)
        tmp += xc[-P2 + k] * c[myid - k];

      for (int k = 0; k < P1; k++)
        tmp += xc[-P1 + k] * d[myid - k];

      xc[s - P4 + myid] = tmp;

    }
    __syncthreads();
  }
}

__global__ void colsGPU(uint32_t *x, int s, int r) {
  int k0 = blockIdx.x * LKNB;     // 
  int k1 = threadIdx.x / LWDR;    // 
  int k2 = threadIdx.x % LWDR;    // 

  __shared__ uint32_t cx[LKNB][2 * P4];

  int fcol = (blockIdx.x == 0) ? 1 : 0;
  int ecol = (blockIdx.x == gridDim.x - 1 && r % LKNB) ? r % LKNB : LKNB;

  for (int i = fcol; i < ecol; i++)
    cx[i][threadIdx.x] = x[(k0 + i) * s - P4 + threadIdx.x];

  __syncthreads();

  uint32_t *pcx = &cx[k1][P4];

  for (int k = 0; k < s / P4 - 1; k++) {

    for (int i = 0; i < P4; i += LWDR)
    {
      if (!(blockIdx.x == 0 && threadIdx.x == 0)
          && !(blockIdx.x == gridDim.x - 1 && k1 >= ecol))
        pcx[i + k2] = pcx[i + k2 - P1] + pcx[i + k2 - P2]
          + pcx[i + k2 - P3] + pcx[i + k2 - P4];

      __syncthreads();

    }

    for (int i = fcol; i < ecol; i++)
      x[(k0 + i) * s + k * P4 + threadIdx.x] = cx[i][threadIdx.x] =
        cx[i][P4 + threadIdx.x];

    __syncthreads();
  }
}

void gLFIB4(uint32_t n, uint32_t *x, int s, int r, uint32_t *seed) {

  cudaMemcpy(x, seed, sizeof(uint32_t) * P4, cudaMemcpyHostToDevice);

  uint32_t *y;
  uint32_t one = 1;

  cudaMalloc((void **) &y, sizeof(uint32_t) * 3 * P4);
  cudaMemset(y + P4 * 2, 0, P4 * sizeof(uint32_t));
  cudaMemcpy(y + P4 * 2, &one, sizeof(uint32_t), cudaMemcpyHostToDevice);

  cudaStream_t cstr1;
  cudaStream_t cstr2;

  cudaStreamCreate(&cstr1);
  cudaStreamCreate(&cstr2);

  firstColGPU<<<1, P4, 0, cstr1>>>(x, s);

  colYGPU<<<1, P4, 0, cstr2>>>(y, s);

  cudaStreamSynchronize(cstr1);
  cudaStreamSynchronize(cstr2);

  lastEntGPU<<<1, 2 * P4>>>(x, y, s, r);
  colsGPU<<<r / LKNB + (r % LKNB ? 1 : 0), P4>>>(x, s, r);

  cudaStreamDestroy(cstr1);
  cudaStreamDestroy(cstr2);

  cudaFree(y);
}

int main(int argc, char**argv) {

  if (argc < 1) {
    printf("Usage: ./main <n>\n");
    return 1;
  }

  uint32_t n = atoi(argv[1]);

  srand(1234);
  uint32_t *x = (uint32_t*) malloc(n * sizeof(uint32_t));

  for (uint32_t r = 16; r <= 4096; r = r * 2) {

    uint32_t s = 0;

    if (s == 0) {
      s = n / r;
      s -= (s % 256 == 0 ? 0 : s % 256);
      while (s * r < n) r++;
    }

    printf("n=%d r=%d s=%d\n", n, r, s);

    uint32_t *z = (uint32_t*) malloc(r * s * sizeof(uint32_t));

    for (uint32_t k = 0; k < P4; k++) x[k] = z[k] = rand();

    // compute on the host
    auto start = std::chrono::steady_clock::now();
    LFIB4(n, x);
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<float> host_time = end - start;

    // compute on the device
    uint32_t *x_d;
    cudaMalloc((void **) &x_d, sizeof(uint32_t) * r * s);

    start = std::chrono::steady_clock::now();
    gLFIB4(n, x_d, s, r, z);
    end = std::chrono::steady_clock::now();
    std::chrono::duration<float> device_time = end - start;
    printf("r = %d | host time = %lf | device time = %lf | speedup = %.1f ",
        r, host_time.count(), device_time.count(), host_time.count() / device_time.count());

    // Verify
    cudaMemcpy(z, x_d, sizeof(uint32_t) * n, cudaMemcpyDeviceToHost);

    bool ok = true;
    for (uint32_t i = 0; i < n; i++) {
      if (x[i] != z[i]) {
        ok = false;
        break;
      }
    }
    printf("check = %s\n", ok ? "PASS" : "FAIL");

    free(z);
    cudaFree(x_d);
  }

  free(x);
  if (!ok) return 1;
  return 0;
}
