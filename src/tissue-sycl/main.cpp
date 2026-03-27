/*
  Accumulate contributions of tissue source
  strengths "qt" and previous solute levels "ctprev" to tissue solute levels "ct".
  Each tissue point is assigned one or more threads: step is the number of threads
  This spreads it over more threads.
  TWS September 2014
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <sycl/sycl.hpp>

void reference(
    const   int *d_tisspoints,
    const float *d_gtt,
    const float *d_gbartt,
          float *d_ct,
    const float *d_ctprev,
    const float *d_qt,
    int nnt, int nntDev, int step, int isp)
{
  for (int i = 0; i < step * nnt; i++) {
    int jtp,ixyz,ix,iy,iz,jx,jy,jz,istep;
    int nnt2 = 2*nnt;
    float p = 0.f;

    int itp = i/step;
    int itp1 = i%step;
    if(itp < nnt) {
      ix = d_tisspoints[itp];
      iy = d_tisspoints[itp+nnt];
      iz = d_tisspoints[itp+nnt2];
      for(jtp=itp1; jtp<nnt; jtp+=step){
        jx = d_tisspoints[jtp];
        jy = d_tisspoints[jtp+nnt];
        jz = d_tisspoints[jtp+nnt2];
        ixyz = abs(jx-ix) + abs(jy-iy) + abs(jz-iz) + (isp-1)*nntDev;
        p += d_gtt[ixyz]*d_ctprev[jtp] + d_gbartt[ixyz]*d_qt[jtp];
      }
      if(itp1 == 0) d_ct[itp] = p;
    }

    for(istep=1; istep<step; istep++)
      if(itp1 == istep && itp < nnt) d_ct[itp] += p;
  }
}

void tissue(
    sycl::nd_item<1> &item,
    const   int *__restrict d_tisspoints,
    const float *__restrict d_gtt,
    const float *__restrict d_gbartt,
          float *__restrict d_ct,
    const float *__restrict d_ctprev,
    const float *__restrict d_qt,
    int nnt, int nntDev, int step, int isp)
{
  const int i = item.get_global_id(0);
  if (i >= step * nnt) return;

  int jtp,ixyz,ix,iy,iz,jx,jy,jz,istep;
  int nnt2 = 2*nnt;
  float p = 0.f;

  const int itp = i/step;
  const int itp1 = i%step;
  if(itp < nnt) {
    ix = d_tisspoints[itp];
    iy = d_tisspoints[itp+nnt];
    iz = d_tisspoints[itp+nnt2];
    for(jtp = itp1; jtp < nnt; jtp += step) {
      jx = d_tisspoints[jtp];
      jy = d_tisspoints[jtp+nnt];
      jz = d_tisspoints[jtp+nnt2];
      ixyz = sycl::abs(jx-ix) + sycl::abs(jy-iy) + sycl::abs(jz-iz) + (isp-1)*nntDev;
      p += d_gtt[ixyz]*d_ctprev[jtp] + d_gbartt[ixyz]*d_qt[jtp];
    }
    if(itp1 == 0) d_ct[itp] = p;
  }
  // d_ct is incremented in sequence from the needed threads
  for(istep=1; istep<step; istep++)
    if(itp1 == istep && itp < nnt) d_ct[itp] += p;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    printf("Usage: %s <dimension of a 3D grid> <repeat>\n", argv[0]);
    return 1;
  }

  const int dim = atoi(argv[1]);
  if (dim > 32) {
    printf("Maximum dimension is 32\n");
    return 1;
  }
  const int repeat = atoi(argv[2]);

  const int nnt = dim * dim * dim;
  const int nntDev = 32*32*32;  // maximum number of tissue points
  const int nsp = 2;

    int* h_tisspoints = (int*) malloc (3*nntDev*sizeof(int));
  float* h_gtt = (float*) malloc (nsp*nntDev*sizeof(float));
  float* h_gbartt = (float*) malloc (nsp*nntDev*sizeof(float));
  float* h_ct = (float*) malloc (nntDev*sizeof(float));
  float* h_ctprev = (float*) malloc (nntDev*sizeof(float));
  float* h_qt = (float*) malloc (nntDev*sizeof(float));
  float* h_ct_gold = (float*) malloc (nntDev*sizeof(float));

  // bound the distance between any two 3D points
  for (int i = 0; i < 3 * nntDev; i++) {
    h_tisspoints[i] = rand() % (nntDev / 3);
  }
  for (int i = 0; i < nsp * nntDev; i++) {
    h_gtt[i] = rand() / (float)RAND_MAX;
    h_gbartt[i] = rand() / (float)RAND_MAX;
  }
  for (int i = 0; i < nntDev; i++) {
    h_ct[i] = h_ct_gold[i] = 0;
    h_ctprev[i] = rand() / (float)RAND_MAX;
    h_qt[i] = rand() / (float)RAND_MAX;
  }

  int step = 4; //a power of two

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

    int *d_tisspoints = sycl::malloc_device<int>(3*nntDev, q);
  float *d_gtt = sycl::malloc_device<float>(nsp*nntDev, q);
  float *d_gbartt = sycl::malloc_device<float>(nsp*nntDev, q);
  float *d_ct = sycl::malloc_device<float>(nntDev, q);
  float *d_ctprev = sycl::malloc_device<float>(nntDev, q);
  float *d_qt = sycl::malloc_device<float>(nntDev, q);

  q.memcpy(d_tisspoints, h_tisspoints, 3*nntDev*sizeof(int));
  q.memcpy(d_gtt, h_gtt, nsp*nntDev*sizeof(float));
  q.memcpy(d_gbartt, h_gbartt, nsp*nntDev*sizeof(float));
  q.memcpy(d_ct, h_ct, nntDev*sizeof(float));
  q.memcpy(d_ctprev, h_ctprev, nntDev*sizeof(float));
  q.memcpy(d_qt, h_qt, nntDev*sizeof(float));

  sycl::range<1> lws (256);
  sycl::range<1> gws ((step*nnt + 255) / 256 * 256);

  auto kFn = [&] (sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      tissue(item, d_tisspoints, d_gtt, d_gbartt, d_ct, d_ctprev, d_qt,
             nnt, nntDev, step, 2);
    });
  };

  // quick verification and warmup
  for (int i = 0; i < 2; i++) {
    q.submit(kFn);
  }

  for (int i = 0; i < 2; i++) {
    reference(h_tisspoints,h_gtt,h_gbartt,h_ct_gold,h_ctprev,h_qt,nnt,nntDev,step,2);
  }

  bool ok = true;
  q.memcpy(h_ct, d_ct, nntDev*sizeof(float)).wait();

  for (int i = 0; i < nntDev; i++) {
    if (fabsf(h_ct[i] - h_ct_gold[i]) > 1e-2) {
      printf("@%d: %f %f\n", i, h_ct[i], h_ct_gold[i]);
      ok = false;
      break;
    }
  }

  printf("%s\n", ok ? "PASS" : "FAIL");

  // timing kernel execution
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit(kFn);
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / repeat);

  free(h_tisspoints);
  free(h_gtt);
  free(h_gbartt);
  free(h_ct);
  free(h_ct_gold);
  free(h_ctprev);
  free(h_qt);
  sycl::free(d_tisspoints, q);
  sycl::free(d_gtt, q);
  sycl::free(d_gbartt, q);
  sycl::free(d_ct, q);
  sycl::free(d_ctprev, q);
  sycl::free(d_qt, q);

  return 0;
}
