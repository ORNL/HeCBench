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
#include <omp.h>
#include "WKFUtils.h"

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
}

void run_gpu_kernel(
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

#pragma omp target data map(to: ax[0:natom], \
                                ay[0:natom], \
                                az[0:natom],\
                                charge[0:natom],\
                                size[0:natom],\
                                gx[0:ngadj],\
                                gy[0:ngadj],\
                                gz[0:ngadj]) \
                        map(alloc: val[0:ngadj])
  {
    wkf_timer_start(timer); 

    for(int n = 0; n < itmax; n++) {
      #pragma omp target teams distribute thread_limit(wgsize)
      for(int igrid=0;igrid<ngrid;igrid++){
        float sum = 0.0f;
        float l_gx = gx[igrid];
        float l_gy = gy[igrid];
        float l_gz = gz[igrid];
        #pragma omp parallel for reduction(+:sum)
        for(int iatom=0; iatom<natom; iatom++) {
          float dist = sqrtf((l_gx-ax[iatom])*(l_gx-ax[iatom]) + 
                             (l_gy-ay[iatom])*(l_gy-ay[iatom]) + 
                             (l_gz-az[iatom])*(l_gz-az[iatom]));

          sum += pre1*(charge[iatom]/dist)*expf(-xkappa*(dist-size[iatom]))
                 / (1.f + xkappa*size[iatom]);
        }
        val[igrid] = sum;
      }
    }

    wkf_timer_stop(timer);
    double avg_kernel_time = wkf_timer_time(timer) / ((double) itmax);
    printf("Average kernel time on the device: %1.12g\n", avg_kernel_time);

    // read output image
    #pragma omp target update from (val[0:ngrid])
  }

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
      float l_gx = gx[igrid];
      float l_gy = gy[igrid];
      float l_gz = gz[igrid];
      #pragma omp parallel for simd reduction(+:sum)
      for(int iatom=0; iatom<natom; iatom++) {
        float dist = sqrtf((l_gx-ax[iatom])*(l_gx-ax[iatom]) + 
                           (l_gy-ay[iatom])*(l_gy-ay[iatom]) + 
                           (l_gz-az[iatom])*(l_gz-az[iatom]));
        sum += pre1*(charge[iatom]/dist)*expf(-xkappa*(dist-size[iatom]))
               / (1.f + xkappa*size[iatom]);
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

  wkf_timer_start(timer);
  run_gpu_kernel(wgsize, itmax, ngrid, natom, ngadj, ax, ay, az, gx, gy, gz, 
                 charge, size, xkappa, pre1, val_gpu);
  wkf_timer_stop(timer);

  printf("GPU Time: %1.12g (Number of tests = %d)\n", wkf_timer_time(timer), itmax);
  SEP;

  compare(val_cpu, val_gpu, ngrid);

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

  if (!ok) return 1;
  return 0;
}
