/*
 * A set of simple Multiple Debye-Huckel (MDH) kernels 
 * inspired by APBS:
 *   http://www.poissonboltzmann.org/ 
 *
 * This code was all originally written by David Gohara on MacOS X, 
 * and has been subsequently been modified by John Stone, porting to Linux,
 * adding vectorization, and several other platform-specific 
 * performance optimizations.
 * 
 */

#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <cuda.h>
#include "WKFUtils.h"
#include "helper_math.h"

#define SEP printf("\n")

void gendata(float *ax,float *ay,float *az,
    float *gx,float *gy,float *gz,
    float *charge,float *size,int natom,int ngrid) {

  int i;    

  printf("Generating Data.. \n");
  for (i=0; i<natom; i++) {
    ax[i] = ((float) rand() / (float) RAND_MAX);
    ay[i] = ((float) rand() / (float) RAND_MAX);
    az[i] = ((float) rand() / (float) RAND_MAX);
    charge[i] = ((float) rand() / (float) RAND_MAX);
    size[i] = (float)natom;
  }

  for (i=0; i<ngrid; i++) {
    gx[i] = ((float) rand() / (float) RAND_MAX);
    gy[i] = ((float) rand() / (float) RAND_MAX);
    gz[i] = ((float) rand() / (float) RAND_MAX);
  }
  printf("Done generating inputs.\n\n");
}

void compare(const float *arr, const float *arr2, int ngrid) {
  int i;
  bool ok = true;
  for (i=0; i<ngrid; i++) {
    if (fabsf(arr[i] - arr2[i]) > 1e-3f) {
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);
}

__global__ void mdh (
    const float *__restrict__ ax, 
    const float *__restrict__ ay,
    const float *__restrict__ az,
    const float *__restrict__ gx, 
    const float *__restrict__ gy, 
    const float *__restrict__ gz,
    const float *__restrict__ charge, 
    const float *__restrict__ size, 
          float *__restrict__ val,
    const float pre1,
    const float xkappa, 
    const int natom)
{
  extern __shared__ float shared[];

  int lid = threadIdx.x;
  int lsize = blockDim.x;
  int igrid = blockIdx.x * lsize + lid;
  float4 v = make_float4(0.0f);
  float4 lgx = reinterpret_cast<const float4*>(gx)[igrid];
  float4 lgy = reinterpret_cast<const float4*>(gy)[igrid];
  float4 lgz = reinterpret_cast<const float4*>(gz)[igrid];

  for(int jatom = 0; jatom < natom; jatom+=lsize )
  {
    if((jatom+lsize) > natom) lsize = natom - jatom;

    if((jatom + lid) < natom) {
      shared[lid * 5    ] = ax[jatom + lid];
      shared[lid * 5 + 1] = ay[jatom + lid];
      shared[lid * 5 + 2] = az[jatom + lid];
      shared[lid * 5 + 3] = charge[jatom + lid];
      shared[lid * 5 + 4] = size[jatom + lid];
    }
    __syncthreads();

    for(int i=0; i<lsize; i++) {
      float4 dx = lgx - shared[i * 5    ];
      float4 dy = lgy - shared[i * 5 + 1];
      float4 dz = lgz - shared[i * 5 + 2];
      float4 dist = sqrtf( dx * dx + dy * dy + dz * dz );
      v += pre1 * (shared[i * 5 + 3] / dist)  *
           expf( -xkappa * (dist - shared[i * 5 + 4])) /
           (1.0f + xkappa * shared[i * 5 + 4]);
    }
    __syncthreads();
  }
  reinterpret_cast<float4*>(val)[ igrid ] = v;
}

__global__ void mdh2 (
    const float *__restrict__ ax, 
    const float *__restrict__ ay,
    const float *__restrict__ az,
    const float *__restrict__ gx, 
    const float *__restrict__ gy, 
    const float *__restrict__ gz,
    const float *__restrict__ charge, 
    const float *__restrict__ size, 
          float *__restrict__ val,
    const float pre1, 
    const float xkappa, 
    const int natom)
{
  extern __shared__ float shared[];

  int lid = threadIdx.x;
  int lsize = blockDim.x;
  int igrid = blockIdx.x * lsize + lid;
  float4 v = make_float4(0.0f);
  float4 lgx = reinterpret_cast<const float4*>(gx)[igrid];
  float4 lgy = reinterpret_cast<const float4*>(gy)[igrid];
  float4 lgz = reinterpret_cast<const float4*>(gz)[igrid];

  for(int jatom = 0; jatom < natom; jatom+=lsize )
  {
    if((jatom+lsize) > natom) lsize = natom - jatom;

    if((jatom + lid) < natom) {
      shared[lid          ] = ax[jatom + lid];
      shared[lid +   lsize] = ay[jatom + lid];
      shared[lid + 2*lsize] = az[jatom + lid];
      shared[lid + 3*lsize] = charge[jatom + lid];
      shared[lid + 4*lsize] = size[jatom + lid];
    }
    __syncthreads();

    for(int i=0; i<lsize; i++) {
      float4 dx = lgx - shared[i          ];
      float4 dy = lgy - shared[i +   lsize];
      float4 dz = lgz - shared[i + 2*lsize];
      float4 dist = sqrtf( dx * dx + dy * dy + dz * dz );
      v += pre1 * ( shared[i + 3*lsize] / dist )  *
           expf( -xkappa * (dist - shared[i + 4*lsize])) /
           (1.0f + xkappa * shared[i + 4*lsize]);
    }
    __syncthreads();
  }
  reinterpret_cast<float4*>(val)[ igrid ] = v;
}

__global__ void mdh3 (
    const float *__restrict__ ax, 
    const float *__restrict__ ay,
    const float *__restrict__ az,
    const float *__restrict__ gx, 
    const float *__restrict__ gy, 
    const float *__restrict__ gz,
    const float *__restrict__ charge, 
    const float *__restrict__ size, 
          float *__restrict__ val,
    const float pre1, 
    const float xkappa, 
    const int natom)
{
  extern __shared__ float shared[];

  int lid = threadIdx.x;
  int lsize = blockDim.x;
  int igrid = blockIdx.x * lsize + lid;
  float4 v = make_float4(0.0f);
  float4 lgx = reinterpret_cast<const float4*>(gx)[igrid];
  float4 lgy = reinterpret_cast<const float4*>(gy)[igrid];
  float4 lgz = reinterpret_cast<const float4*>(gz)[igrid];

  for(int jatom = 0; jatom < natom; jatom+=lsize )
  {
    if((jatom+lsize) > natom) lsize = natom - jatom;

    if((jatom + lid) < natom) {
      shared[lid          ] = ax[jatom + lid];
      shared[lid +   lsize] = ay[jatom + lid];
      shared[lid + 2*lsize] = az[jatom + lid];
      shared[lid + 3*lsize] = charge[jatom + lid];
      shared[lid + 4*lsize] = size[jatom + lid];
    }
    __syncthreads();

    for(int i=0; i<lsize; i++) {
      float4 dx = lgx - shared[i          ];
      float4 dy = lgy - shared[i +   lsize];
      float4 dz = lgz - shared[i + 2*lsize];
      float4 dist = fast_sqrtf( dx * dx + dy * dy + dz * dz );
      v += pre1 * ( shared[i + 3*lsize] / dist )  *
           fast_expf( -xkappa * (dist - shared[i + 4*lsize])) /
           (1.0f + xkappa * shared[i + 4*lsize]);
    }
    __syncthreads();
  }
  reinterpret_cast<float4*>(val)[ igrid ] = v;
}

void run_gpu_kernel(
    const int choice, 
    const int wgsize, 
    const int itmax,
    const int ngrid,
    const int natom,
    const int ngadj,
    const float *ax, 
    const float *ay,
    const float *az,
    const float *gx, 
    const float *gy, 
    const float *gz,
    const float *charge, 
    const float *size, 
    const float xkappa, 
    const float pre1, 
          float *val)
{
  wkf_timerhandle timer = wkf_timer_create();

  //Allocate memory for programs and kernels
  float* d_ax;
  cudaMalloc((void**)&d_ax, sizeof(float)*natom);
  cudaMemcpy(d_ax, ax, sizeof(float)*natom, cudaMemcpyHostToDevice);

  float* d_ay;
  cudaMalloc((void**)&d_ay, sizeof(float)*natom);
  cudaMemcpy(d_ay, ay, sizeof(float)*natom, cudaMemcpyHostToDevice);

  float* d_az;
  cudaMalloc((void**)&d_az, sizeof(float)*natom);
  cudaMemcpy(d_az, az, sizeof(float)*natom, cudaMemcpyHostToDevice);

  float* d_charge;
  cudaMalloc((void**)&d_charge, sizeof(float)*natom);
  cudaMemcpy(d_charge, charge, sizeof(float)*natom, cudaMemcpyHostToDevice);

  float* d_size;
  cudaMalloc((void**)&d_size, sizeof(float)*natom);
  cudaMemcpy(d_size, size, sizeof(float)*natom, cudaMemcpyHostToDevice);

  float* d_gx;
  cudaMalloc((void**)&d_gx, sizeof(float)*ngadj);
  cudaMemcpy(d_gx, gx, sizeof(float)*ngadj, cudaMemcpyHostToDevice);

  float* d_gy;
  cudaMalloc((void**)&d_gy, sizeof(float)*ngadj);
  cudaMemcpy(d_gy, gy, sizeof(float)*ngadj, cudaMemcpyHostToDevice);

  float* d_gz;
  cudaMalloc((void**)&d_gz, sizeof(float)*ngadj);
  cudaMemcpy(d_gz, gz, sizeof(float)*ngadj, cudaMemcpyHostToDevice);

  float* d_val;
  cudaMalloc((void**)&d_val, sizeof(float)*ngadj);

  wkf_timer_start(timer); 

  // set work-item dimensions 
  // scale number of work units by vector size
  dim3 grids (ngadj / 4 / wgsize);
  dim3 blocks (wgsize);
  const int sm_size = sizeof(float) * 5 * wgsize;

  for(int n = 0; n < itmax; n++) {
    if (choice == 0)
      mdh<<<grids, blocks, sm_size>>>(
          d_ax,
          d_ay,
          d_az,
          d_gx,
          d_gy,
          d_gz,
          d_charge,
          d_size,
          d_val,
          pre1,
          xkappa,
          natom);
    else if (choice == 1)
      mdh2<<<grids, blocks, sm_size>>>(
          d_ax,
          d_ay,
          d_az,
          d_gx,
          d_gy,
          d_gz,
          d_charge,
          d_size,
          d_val,
          pre1,
          xkappa,
          natom);
    else
      mdh3<<<grids, blocks, sm_size>>>(
          d_ax,
          d_ay,
          d_az,
          d_gx,
          d_gy,
          d_gz,
          d_charge,
          d_size,
          d_val,
          pre1,
          xkappa,
          natom);
  }
  cudaDeviceSynchronize();

  wkf_timer_stop(timer);
  double avg_kernel_time = wkf_timer_time(timer) / ((double) itmax);
  printf("Average kernel execution time: %1.12g\n", avg_kernel_time);

  // read output image
  cudaMemcpy(val, d_val, sizeof(float)*ngrid, cudaMemcpyDeviceToHost);

  cudaFree(d_ax);
  cudaFree(d_ay);
  cudaFree(d_az);
  cudaFree(d_gx);
  cudaFree(d_gy);
  cudaFree(d_gz);
  cudaFree(d_val);
  cudaFree(d_size);
  cudaFree(d_charge);

  wkf_timer_destroy(timer);
}

// reference CPU kernel
void run_cpu_kernel(
    const int itmax,
    const int ngrid,
    const int natom,
    const float *ax,
    const float *ay,
    const float *az,
    const float *gx,
    const float *gy,
    const float *gz,
    const float *charge,
    const float *size,
    const float xkappa,
    const float pre1,
          float *val)
{
  wkf_timerhandle timer = wkf_timer_create();
  wkf_timer_start(timer); 

  for(int n = 0; n < itmax; n++) {
    #pragma omp parallel for
    for(int igrid=0;igrid<ngrid;igrid++){
      float sum = 0.0f;
      #pragma omp parallel for simd reduction(+:sum)
      for(int iatom=0; iatom<natom; iatom++) {
        float dist = sqrtf((gx[igrid]-ax[iatom])*(gx[igrid]-ax[iatom]) + 
            (gy[igrid]-ay[iatom])*(gy[igrid]-ay[iatom]) + 
            (gz[igrid]-az[iatom])*(gz[igrid]-az[iatom]));

        sum += pre1*(charge[iatom]/dist)*expf(-xkappa*(dist-size[iatom]))
          / (1+xkappa*size[iatom]);
      }
      val[igrid] = sum;
    }
  }

  wkf_timer_stop(timer);
  double avg_kernel_time = wkf_timer_time(timer) / ((double) itmax);
  printf("Average kernel execution time: %1.12g\n", avg_kernel_time);

  wkf_timer_destroy(timer);
}


void usage() {
  printf("command line parameters:\n");
  printf("Optional test flags:\n");
  printf("  -itmax N         loop test N times\n");
  printf("  -wgsize          set workgroup size\n");
}

void getargs(int argc, const char **argv, int *itmax, int *wgsize) {
  int i;
  for (i=0; i<argc; i++) {
    if ((!strcmp(argv[i], "-itmax")) && ((i+1) < argc)) {
      i++;
      *itmax = atoi(argv[i]);
    }

    if ((!strcmp(argv[i], "-wgsize")) && ((i+1) < argc)) {
      i++;
      *wgsize = atoi(argv[i]);
    }
  }

  printf("Run parameters:\n");
  printf("  kernel loop count: %d\n", *itmax);
  printf("     workgroup size: %d\n", *wgsize);
}


int main(int argc, const char **argv) {
  int itmax = 100;
  int wgsize = 256;

  getargs(argc, argv, &itmax, &wgsize);

  wkf_timerhandle timer = wkf_timer_create();

  int natom = 5877;
  int ngrid = 134918;
  int ngadj = ngrid + (512 - (ngrid & 511));

  float pre1 = 4.46184985145e19;
  float xkappa = 0.0735516324639;

  float *ax = (float*)calloc(natom, sizeof(float));
  float *ay = (float*)calloc(natom, sizeof(float));
  float *az = (float*)calloc(natom, sizeof(float));
  float *charge = (float*)calloc(natom, sizeof(float));
  float *size = (float*)calloc(natom, sizeof(float));

  float *gx = (float*)calloc(ngadj, sizeof(float));
  float *gy = (float*)calloc(ngadj, sizeof(float));
  float *gz = (float*)calloc(ngadj, sizeof(float));

  // result
  float *val_cpu = (float*)calloc(ngadj, sizeof(float));
  float *val_gpu = (float*)calloc(ngadj, sizeof(float));

  gendata(ax, ay, az, gx, gy, gz, charge, size, natom, ngrid);

  wkf_timer_start(timer);
  run_cpu_kernel(itmax, ngadj, natom, ax, ay, az, gx, gy, gz, charge, size, xkappa, pre1, val_cpu);
  wkf_timer_stop(timer);

  printf("CPU Time: %1.12g (Number of tests = %d)\n", wkf_timer_time(timer), itmax);
  SEP;

  for (int choice = 0; choice < 3; choice++) {
    wkf_timer_start(timer);
    run_gpu_kernel(choice, wgsize, itmax, ngrid, natom, ngadj, ax, ay, az, gx, gy, gz, 
                   charge, size, xkappa, pre1, val_gpu);
    wkf_timer_stop(timer);

    printf("GPU Time: %1.12g (Number of tests = %d)\n", wkf_timer_time(timer), itmax);
    SEP;
    compare(val_cpu, val_gpu, ngrid);
  }

  free(ax);
  free(ay);
  free(az);
  free(charge);
  free(size);
  free(gx);
  free(gy);
  free(gz);
  free(val_cpu);
  free(val_gpu);

  wkf_timer_destroy(timer);

  return 0;
}
