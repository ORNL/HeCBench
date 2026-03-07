#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <cuda.h>

__global__ void
haccmk_kernel (
    const int n1,  // outer loop count
    const int n2,  // inner loop count
    const float *__restrict__ xx, 
    const float *__restrict__ yy,
    const float *__restrict__ zz,
    const float *__restrict__ mass,
          float *__restrict__ vx2,
          float *__restrict__ vy2,
          float *__restrict__ vz2,
    const float fsrmax,
    const float mp_rsm,
    const float fcoeff ) 
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n1) return;

  const float ma0 = 0.269327f; 
  const float ma1 = -0.0750978f; 
  const float ma2 = 0.0114808f; 
  const float ma3 = -0.00109313f; 
  const float ma4 = 0.0000605491f; 
  const float ma5 = -0.00000147177f;

  float dxc, dyc, dzc, m, r2, f, xi, yi, zi;

  xi = 0.f; 
  yi = 0.f;
  zi = 0.f;

  float xxi = xx[i];
  float yyi = yy[i];
  float zzi = zz[i];

  for ( int j = 0; j < n2; j++ ) {
    dxc = xx[j] - xxi;
    dyc = yy[j] - yyi;
    dzc = zz[j] - zzi;

    r2 = dxc * dxc + dyc * dyc + dzc * dzc;

    //if ( r2 < fsrmax ) m = mass[j]; else m = 0.f;
    m = mass[j] * (r2 < fsrmax);

    f = r2 + mp_rsm;
    f = m * (1.f / (f * sqrtf(f)) - (ma0 + r2*(ma1 + r2*(ma2 + r2*(ma3 + r2*(ma4 + r2*ma5))))));

    xi = xi + f * dxc;
    yi = yi + f * dyc;
    zi = zi + f * dzc;
  }

  vx2[i] += xi * fcoeff;
  vy2[i] += yi * fcoeff;
  vz2[i] += zi * fcoeff;
}

void haccmk (
    const int repeat,
    const int n1,
    const int n2,
    const float* xx, 
    const float* yy,
    const float* zz,
    const float* mass,
    float* vx2,
    float* vy2,
    float* vz2,
    const float fsrmax,
    const float mp_rsm,
    const float fcoeff ) 
{
  const int block_size = 256;

  float *d_xx, *d_yy, *d_zz, *d_mass;
  float *d_vx2, *d_vy2, *d_vz2;

  cudaMalloc((void**) &d_xx, sizeof(float) * n2);
  cudaMalloc((void**) &d_yy, sizeof(float) * n2);
  cudaMalloc((void**) &d_zz, sizeof(float) * n2);
  cudaMalloc((void**) &d_mass, sizeof(float) * n2);
  cudaMalloc((void**) &d_vx2, sizeof(float) * n1);
  cudaMalloc((void**) &d_vy2, sizeof(float) * n1);
  cudaMalloc((void**) &d_vz2, sizeof(float) * n1);

  cudaMemcpy(d_xx, xx, sizeof(float) * n2, cudaMemcpyHostToDevice);
  cudaMemcpy(d_yy, yy, sizeof(float) * n2, cudaMemcpyHostToDevice);
  cudaMemcpy(d_zz, zz, sizeof(float) * n2, cudaMemcpyHostToDevice);
  cudaMemcpy(d_mass, mass, sizeof(float) * n2, cudaMemcpyHostToDevice);

  dim3 grids ((n1+block_size-1)/block_size);
  dim3 blocks (block_size);

  float total_time = 0.f;

  for (int i = 0; i < repeat; i++) {
    // reset output
    cudaMemcpy(d_vx2, vx2, sizeof(float) * n1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy2, vy2, sizeof(float) * n1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz2, vz2, sizeof(float) * n1, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    haccmk_kernel <<< grids, blocks >>> (
      n1, n2, d_xx, d_yy, d_zz, d_mass,
      d_vx2, d_vy2, d_vz2, fsrmax, mp_rsm, fcoeff);

    cudaDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    total_time += time;
  }

  printf("Average kernel execution time %f (s)\n", (total_time * 1e-9f) / repeat);

  cudaMemcpy(vx2, d_vx2, sizeof(float) * n1, cudaMemcpyDeviceToHost);
  cudaMemcpy(vy2, d_vy2, sizeof(float) * n1, cudaMemcpyDeviceToHost);
  cudaMemcpy(vz2, d_vz2, sizeof(float) * n1, cudaMemcpyDeviceToHost);
  cudaFree(d_xx);
  cudaFree(d_yy);
  cudaFree(d_zz);
  cudaFree(d_mass);
  cudaFree(d_vx2);
  cudaFree(d_vy2);
  cudaFree(d_vz2);
}

void haccmk_gold(
    int count1,
    float xxi,
    float yyi,
    float zzi,
    float fsrrmax2,
    float mp_rsm2, 
    float *__restrict__ xx1, 
    float *__restrict__ yy1, 
    float *__restrict__ zz1, 
    float *__restrict__ mass1, 
    float *__restrict__ dxi,
    float *__restrict__ dyi,
    float *__restrict__ dzi )
{
  int j;
  const float ma0 = 0.269327, ma1 = -0.0750978, ma2 = 0.0114808, 
        ma3 = -0.00109313, ma4 = 0.0000605491, ma5 = -0.00000147177;
  float dxc, dyc, dzc, m, r2, f, xi, yi, zi;

  xi = 0.f; 
  yi = 0.f;
  zi = 0.f;

  for ( j = 0; j < count1; j++ ) {
    dxc = xx1[j] - xxi;
    dyc = yy1[j] - yyi;
    dzc = zz1[j] - zzi;

    r2 = dxc * dxc + dyc * dyc + dzc * dzc;

    if ( r2 < fsrrmax2 ) m = mass1[j]; else m = 0.f;

    f = r2 + mp_rsm2;
    f =  m * ( 1.f / ( f * sqrtf( f ) ) - ( ma0 + r2*(ma1 + r2*(ma2 + r2*(ma3 + r2*(ma4 + r2*ma5))))));

    xi = xi + f * dxc;
    yi = yi + f * dyc;
    zi = zi + f * dzc;
  }

  *dxi = xi;
  *dyi = yi;
  *dzi = zi;
}

int main( int argc, char *argv[] )
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  float fsrrmax2, mp_rsm2, fcoeff, dx1, dy1, dz1, dx2, dy2, dz2;
  int n1, n2, i;
  n1 = 784;
  n2 = 15000;
  printf( "Outer loop count is set %d\n", n1 );
  printf( "Inner loop count is set %d\n", n2 );

  float* xx = (float*) malloc (sizeof(float) * n2);
  float* yy = (float*) malloc (sizeof(float) * n2);
  float* zz = (float*) malloc (sizeof(float) * n2);
  float* mass = (float*) malloc (sizeof(float) * n2);
  float* vx2 = (float*) malloc (sizeof(float) * n2);
  float* vy2 = (float*) malloc (sizeof(float) * n2);
  float* vz2 = (float*) malloc (sizeof(float) * n2);
  float* vx2_hw = (float*) malloc (sizeof(float) * n2);
  float* vy2_hw = (float*) malloc (sizeof(float) * n2);
  float* vz2_hw = (float*) malloc (sizeof(float) * n2);

  /* Initial data preparation */
  fcoeff = 0.23f;  
  fsrrmax2 = 0.5f; 
  mp_rsm2 = 0.03f;
  dx1 = 1.0f/(float)n2;
  dy1 = 2.0f/(float)n2;
  dz1 = 3.0f/(float)n2;
  xx[0] = 0.f;
  yy[0] = 0.f;
  zz[0] = 0.f;
  mass[0] = 2.f;

  for ( i = 1; i < n2; i++ ) {
    xx[i] = xx[i-1] + dx1;
    yy[i] = yy[i-1] + dy1;
    zz[i] = zz[i-1] + dz1;
    mass[i] = (float)i * 0.01f + xx[i];
  }

  for ( i = 0; i < n2; i++ ) {
    vx2[i] = 0.f;
    vy2[i] = 0.f;
    vz2[i] = 0.f;
    vx2_hw[i] = 0.f;
    vy2_hw[i] = 0.f;
    vz2_hw[i] = 0.f;
  }

  for ( i = 0; i < n1; ++i) {
    haccmk_gold( n2, xx[i], yy[i], zz[i], fsrrmax2, mp_rsm2, xx, yy, zz, mass, &dx2, &dy2, &dz2 );    
    vx2[i] = vx2[i] + dx2 * fcoeff;
    vy2[i] = vy2[i] + dy2 * fcoeff;
    vz2[i] = vz2[i] + dz2 * fcoeff;
  }

  haccmk(repeat, n1, n2, xx, yy, zz, mass,
         vx2_hw, vy2_hw, vz2_hw, fsrrmax2, mp_rsm2, fcoeff);

  // verify
  int error = 0;
  const float eps = 1.0f;
  for (i = 0; i < n2; i++) {
    if (fabsf(vx2[i] - vx2_hw[i]) > eps) {
      printf("error at vx2[%d] %f %f\n", i, vx2[i], vx2_hw[i]);
      error = 1;
      break;
    }
    if (fabsf(vy2[i] - vy2_hw[i]) > eps) {
      printf("error at vy2[%d]: %f %f\n", i, vy2[i], vy2_hw[i]);
      error = 1;
      break;
    }
    if (fabsf(vz2[i] - vz2_hw[i]) > eps) {
      printf("error at vz2[%d]: %f %f\n", i, vz2[i], vz2_hw[i]);
      error = 1;
      break;
    }
  }

  free(xx);
  free(yy);
  free(zz);
  free(mass);
  free(vx2);
  free(vy2);
  free(vz2);
  free(vx2_hw);
  free(vy2_hw);
  free(vz2_hw);

  printf("%s\n", error ? "FAIL" : "PASS");
  if (error) exit(1);

  return 0;
}
