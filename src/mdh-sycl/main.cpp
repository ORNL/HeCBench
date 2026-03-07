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
#include <sycl/sycl.hpp>
#include "WKFUtils.h"

#define SEP printf("\n")

void gendata(float *ax,float *ay,float *az,
    float *gx,float *gy,float *gz,
    float *charge,float *size,int natom,int ngrid) {

  int i;

  printf("Generating Data..\n");
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
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  wkf_timerhandle timer = wkf_timer_create();

  //Allocate memory for programs and kernels
  float *d_ax = sycl::malloc_device<float>(natom, q);
  q.memcpy(d_ax, ax, sizeof(float)*natom);

  float *d_ay = sycl::malloc_device<float>(natom, q);
  q.memcpy(d_ay, ay, sizeof(float)*natom);

  float *d_az = sycl::malloc_device<float>(natom, q);
  q.memcpy(d_az, az, sizeof(float)*natom);

  float *d_charge = sycl::malloc_device<float>(natom, q);
  q.memcpy(d_charge, charge, sizeof(float)*natom);

  float *d_size = sycl::malloc_device<float>(natom, q);
  q.memcpy(d_size, size, sizeof(float)*natom);

  float *d_gx = sycl::malloc_device<float>(ngadj, q);
  q.memcpy(d_gx, gx, sizeof(float)*ngadj);

  float *d_gy = sycl::malloc_device<float>(ngadj, q);
  q.memcpy(d_gy, gy, sizeof(float)*ngadj);

  float *d_gz = sycl::malloc_device<float>(ngadj, q);
  q.memcpy(d_gz, gz, sizeof(float)*ngadj);

  float *d_val = sycl::malloc_device<float>(ngadj, q);
  q.wait();

  wkf_timer_start(timer);

  // set work-item dimensions
  // scale number of work units by vector size
  sycl::range<1> gws (ngadj / 4);
  sycl::range<1> lws (wgsize);

  for(int n = 0; n < itmax; n++) {
    if (choice == 0)
      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<float> shared(sycl::range<1>(5*wgsize), cgh);
        cgh.parallel_for<class mdh_v4>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          int igrid = item.get_global_id(0);
          int lsize = item.get_local_range(0);
          int lid = item.get_local_id(0);
          sycl::float4 v (0.0f);
          sycl::float4 lgx = reinterpret_cast<const sycl::float4*>(d_gx)[igrid];
          sycl::float4 lgy = reinterpret_cast<const sycl::float4*>(d_gy)[igrid];
          sycl::float4 lgz = reinterpret_cast<const sycl::float4*>(d_gz)[igrid];

          for(int jatom = 0; jatom < natom; jatom+=lsize )
          {
            if((jatom+lsize) > natom) lsize = natom - jatom;

            if((jatom + lid) < natom) {
              shared[lid * 5    ] = d_ax[jatom + lid];
              shared[lid * 5 + 1] = d_ay[jatom + lid];
              shared[lid * 5 + 2] = d_az[jatom + lid];
              shared[lid * 5 + 3] = d_charge[jatom + lid];
              shared[lid * 5 + 4] = d_size[jatom + lid];
            }
            item.barrier(sycl::access::fence_space::local_space);

            for(int i=0; i<lsize; i++) {
              sycl::float4 dx = lgx - shared[i * 5         ];
              sycl::float4 dy = lgy - shared[i * 5 + 1];
              sycl::float4 dz = lgz - shared[i * 5 + 2];
              sycl::float4 dist = sycl::sqrt(dx*dx + dy*dy + dz*dz);
              v += pre1 * (shared[i * 5 + 3] / dist)  *
                sycl::exp( -xkappa * (dist - shared[i * 5 + 4])) /
                (1.0f + xkappa * shared[i * 5 + 4]);
            }
            item.barrier(sycl::access::fence_space::local_space);
          }
          reinterpret_cast<sycl::float4*>(d_val)[ igrid ] = v;
        });
      });
    else if (choice == 1)
      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<float> shared(sycl::range<1>(5*wgsize), cgh);
        cgh.parallel_for<class mdh2_v4>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          int igrid = item.get_global_id(0);
          int lsize = item.get_local_range(0);
          int lid = item.get_local_id(0);
          sycl::float4 v (0.0f);
          sycl::float4 lgx = reinterpret_cast<const sycl::float4*>(d_gx)[igrid];
          sycl::float4 lgy = reinterpret_cast<const sycl::float4*>(d_gy)[igrid];
          sycl::float4 lgz = reinterpret_cast<const sycl::float4*>(d_gz)[igrid];

          for(int jatom = 0; jatom < natom; jatom+=lsize )
          {
            if((jatom+lsize) > natom) lsize = natom - jatom;

            if((jatom + lid) < natom) {
              shared[lid          ] = d_ax[jatom + lid];
              shared[lid +   lsize] = d_ay[jatom + lid];
              shared[lid + 2*lsize] = d_az[jatom + lid];
              shared[lid + 3*lsize] = d_charge[jatom + lid];
              shared[lid + 4*lsize] = d_size[jatom + lid];
            }
            item.barrier(sycl::access::fence_space::local_space);

            for(int i=0; i<lsize; i++) {
              sycl::float4 dx = lgx - shared[i          ];
              sycl::float4 dy = lgy - shared[i +   lsize];
              sycl::float4 dz = lgz - shared[i + 2*lsize];
              sycl::float4 dist = sycl::sqrt( dx * dx + dy * dy + dz * dz );
              v += pre1 * ( shared[i + 3*lsize] / dist )  *
                sycl::exp( -xkappa * (dist - shared[i + 4*lsize])) /
                (1.0f + xkappa * shared[i + 4*lsize]);

            }
            item.barrier(sycl::access::fence_space::local_space);
          }
          reinterpret_cast<sycl::float4*>(d_val)[ igrid ] = v;
        });
      });
    else
      q.submit([&] (sycl::handler &cgh) {
        sycl::local_accessor<float> shared(sycl::range<1>(5*wgsize), cgh);
        cgh.parallel_for<class mdh3_v4>(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          int igrid = item.get_global_id(0);
          int lsize = item.get_local_range(0);
          int lid = item.get_local_id(0);
          sycl::float4 v (0.0f);
          sycl::float4 lgx = reinterpret_cast<const sycl::float4*>(d_gx)[igrid];
          sycl::float4 lgy = reinterpret_cast<const sycl::float4*>(d_gy)[igrid];
          sycl::float4 lgz = reinterpret_cast<const sycl::float4*>(d_gz)[igrid];

          for(int jatom = 0; jatom < natom; jatom+=lsize )
          {
            if((jatom+lsize) > natom) lsize = natom - jatom;

            if((jatom + lid) < natom) {
              shared[lid          ] = d_ax[jatom + lid];
              shared[lid +   lsize] = d_ay[jatom + lid];
              shared[lid + 2*lsize] = d_az[jatom + lid];
              shared[lid + 3*lsize] = d_charge[jatom + lid];
              shared[lid + 4*lsize] = d_size[jatom + lid];
            }
            item.barrier(sycl::access::fence_space::local_space);

            for(int i=0; i<lsize; i++) {
              sycl::float4 dx = lgx - shared[i          ];
              sycl::float4 dy = lgy - shared[i +   lsize];
              sycl::float4 dz = lgz - shared[i + 2*lsize];
              sycl::float4 dist = sycl::native::sqrt( dx * dx + dy * dy + dz * dz );
              v += pre1 * ( shared[i + 3*lsize] / dist )  *
                   sycl::native::exp( -xkappa * (dist - shared[i + 4*lsize])) /
                   (1.0f + xkappa * shared[i + 4*lsize]);

            }
            item.barrier(sycl::access::fence_space::local_space);
          }
          reinterpret_cast<sycl::float4*>(d_val)[ igrid ] = v;
        });
      });
  }
  q.wait();

  wkf_timer_stop(timer);
  double avg_kernel_time = wkf_timer_time(timer) / ((double) itmax);
  printf("Average kernel execution time: %1.12g\n", avg_kernel_time);

  // read output image
  q.memcpy(val, d_val, sizeof(float)*ngrid).wait();

  sycl::free(d_ax, q);
  sycl::free(d_ay, q);
  sycl::free(d_az, q);
  sycl::free(d_gx, q);
  sycl::free(d_gy, q);
  sycl::free(d_gz, q);
  sycl::free(d_val, q);
  sycl::free(d_size, q);
  sycl::free(d_charge, q);

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
    run_gpu_kernel(true, wgsize, itmax, ngrid, natom, ngadj, ax, ay, az, gx, gy, gz,
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

  if (!ok) return 1;
  return 0;
}
