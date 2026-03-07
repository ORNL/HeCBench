/*
 * GPU-accelerated AIDW interpolation algorithm 
 *
 * Implemented with / without CUDA Shared Memory
 *
 * By Dr.Gang Mei
 *
 * Created on 2015.11.06, China University of Geosciences, 
 *                        gang.mei@cugb.edu.cn
 * Revised on 2015.12.14, China University of Geosciences, 
 *                        gang.mei@cugb.edu.cn
 * 
 * Related publications:
 *  1) "Evaluating the Power of GPU Acceleration for IDW Interpolation Algorithm"
 *     http://www.hindawi.com/journals/tswj/2014/171574/
 *  2) "Accelerating Adaptive IDW Interpolation Algorithm on a Single GPU"
 *     http://arxiv.org/abs/1511.02186
 *
 * License: http://creativecommons.org/licenses/by/4.0/
 */

#include <cstdio>
#include <cstdlib>     
#include <vector>
#include <cmath>
#include <chrono>
#include <sycl/sycl.hpp>
#include "reference.h"

// Calculate the power parameter, and then weighted interpolating
// Without using shared memory 
void AIDW_Kernel(
    const float *__restrict dx, 
    const float *__restrict dy,
    const float *__restrict dz,
    const int dnum,
    const float *__restrict ix,
    const float *__restrict iy,
          float *__restrict iz,
    const int inum,
    const float area,
    const float *__restrict avg_dist,
    sycl::nd_item<1> &item) 

{
  int tid = item.get_global_id(0);
  if (tid >= inum) return;

  float sum = 0.f, dist = 0.f, t = 0.f, z = 0.f, alpha = 1.f;

  float r_obs = avg_dist[tid];                // The observed average nearest neighbor distance
  float r_exp = 0.5f / sycl::sqrt(dnum / area); // The expected nearest neighbor distance for a random pattern
  float R_S0 = r_obs / r_exp;                 // The nearest neighbor statistic

  // Normalize the R(S0) measure such that it is bounded by 0 and 1 by a fuzzy membership function 
  float u_R = 0.f;
  if(R_S0 >= R_min) u_R = 0.5f-0.5f * sycl::cos(3.1415926f / R_max * (R_S0 - R_min));
  if(R_S0 >= R_max) u_R = 1.f;

  // Determine the appropriate distance-decay parameter alpha by a triangular membership function
  // Adaptive power parameter: a (alpha)
  if(u_R>= 0.f && u_R<=0.1f)  alpha = a1; 
  if(u_R>0.1f && u_R<=0.3f)  alpha = a1*(1.f-5.f*(u_R-0.1f)) + a2*5.f*(u_R-0.1f);
  if(u_R>0.3f && u_R<=0.5f)  alpha = a3*5.f*(u_R-0.3f) + a1*(1.f-5.f*(u_R-0.3f));
  if(u_R>0.5f && u_R<=0.7f)  alpha = a3*(1.f-5.f*(u_R-0.5f)) + a4*5.f*(u_R-0.5f);
  if(u_R>0.7f && u_R<=0.9f)  alpha = a5*5.f*(u_R-0.7f) + a4*(1.f-5.f*(u_R-0.7f));
  if(u_R>0.9f && u_R<=1.f)  alpha = a5;
  alpha *= 0.5f; // Half of the power

  // Weighted average
  for(int j = 0; j < dnum; j++) {
    dist = (ix[tid] - dx[j]) * (ix[tid] - dx[j]) + (iy[tid] - dy[j]) * (iy[tid] - dy[j]) ;
    t = 1.f / sycl::pow(dist, alpha);  sum += t;  z += dz[j] * t;
  }
  iz[tid] = z / sum;
}

// Calculate the power parameter, and then weighted interpolating
// With using shared memory (Tiled version of the stage 2)
void AIDW_Kernel_Tiled(
    const float *__restrict dx, 
    const float *__restrict dy,
    const float *__restrict dz,
    const int dnum,
    const float *__restrict ix,
    const float *__restrict iy,
          float *__restrict iz,
    const int inum,
    const float area,
    const float *__restrict avg_dist,
          float *__restrict sdx,
          float *__restrict sdy,
          float *__restrict sdz,
    sycl::nd_item<1> &item) 
{
  int tid = item.get_global_id(0);
  if (tid >= inum) return;

  float dist = 0.f, t = 0.f, alpha = 0.f;

  int part = (dnum - 1) / BLOCK_SIZE;
  int m, e;

  float sum_up = 0.f;
  float sum_dn = 0.f;   
  float six_s, siy_s;

  float r_obs = avg_dist[tid];               //The observed average nearest neighbor distance
  float r_exp = 0.5f / sycl::sqrt(dnum / area); // The expected nearest neighbor distance for a random pattern
  float R_S0 = r_obs / r_exp;                //The nearest neighbor statistic

  float u_R = 0.f;
  if(R_S0 >= R_min) u_R = 0.5f-0.5f * sycl::cos(3.1415926f / R_max * (R_S0 - R_min));
  if(R_S0 >= R_max) u_R = 1.f;

  // Determine the appropriate distance-decay parameter alpha by a triangular membership function
  // Adaptive power parameter: a (alpha)
  if(u_R>= 0.f && u_R<=0.1f)  alpha = a1; 
  if(u_R>0.1f && u_R<=0.3f)  alpha = a1*(1.f-5.f*(u_R-0.1f)) + a2*5.f*(u_R-0.1f);
  if(u_R>0.3f && u_R<=0.5f)  alpha = a3*5.f*(u_R-0.3f) + a1*(1.f-5.f*(u_R-0.3f));
  if(u_R>0.5f && u_R<=0.7f)  alpha = a3*(1.f-5.f*(u_R-0.5f)) + a4*5.f*(u_R-0.5f);
  if(u_R>0.7f && u_R<=0.9f)  alpha = a5*5.f*(u_R-0.7f) + a4*(1.f-5.f*(u_R-0.7f));
  if(u_R>0.9f && u_R<=1.f)  alpha = a5;
  alpha *= 0.5f; // Half of the power

  float six_t = ix[tid];
  float siy_t = iy[tid];
  int lid = item.get_local_id(0);
  for(m = 0; m <= part; m++) {  // Weighted Sum  
    int num_threads = sycl::min(BLOCK_SIZE, dnum - BLOCK_SIZE *m);
    if (lid < num_threads) {
      sdx[lid] = dx[lid + BLOCK_SIZE * m];
      sdy[lid] = dy[lid + BLOCK_SIZE * m];
      sdz[lid] = dz[lid + BLOCK_SIZE * m];
    }
    item.barrier(sycl::access::fence_space::local_space);

    for(e = 0; e < BLOCK_SIZE; e++) {
      six_s = six_t - sdx[e];
      siy_s = siy_t - sdy[e];
      dist = (six_s * six_s + siy_s * siy_s);
      t = 1.f / (sycl::pow(dist, alpha));  sum_dn += t;  sum_up += t * sdz[e];
    }
    item.barrier(sycl::access::fence_space::local_space);
  }
  iz[tid] = sum_up / sum_dn;
}

int main(int argc, char *argv[])
{
  if (argc != 4) {
    printf("Usage: %s <pts> <check> <iterations>\n", argv[0]);
    printf("pts: number of points (unit: 1K)\n");
    printf("check: enable verification when the value is 1\n");
    return 1;
  }

  const int numk = atoi(argv[1]); // number of points (unit: 1K)
  const int check = atoi(argv[2]); // do check for small problem sizes
  const int iterations = atoi(argv[3]); // repeat kernel execution

  const int dnum = numk * 1024;
  const int inum = dnum;

  // Area of planar region
  const float width = 2000, height = 2000;
  const float area = width * height;

  std::vector<float> dx(dnum), dy(dnum), dz(dnum);
  std::vector<float> avg_dist(dnum);
  std::vector<float> ix(inum), iy(inum), iz(inum);
  std::vector<float> h_iz(inum);

  srand(123);
  for(int i = 0; i < dnum; i++)
  {
    dx[i] = rand()/(float)RAND_MAX * 1000;
    dy[i] = rand()/(float)RAND_MAX * 1000;
    dz[i] = rand()/(float)RAND_MAX * 1000;
  }

  for(int i = 0; i < inum; i++)
  {
    ix[i] = rand()/(float)RAND_MAX * 1000;
    iy[i] = rand()/(float)RAND_MAX * 1000;
    iz[i] = 0.f;
  }

  for(int i = 0; i < dnum; i++)
  {
    avg_dist[i] = rand()/(float)RAND_MAX * 3;
  }

  printf("Size = : %d K \n", numk);
  printf("dnum = : %d\ninum = : %d\n", dnum, inum);

  if (check) {
    printf("Verification enabled\n");
    reference (dx.data(), dy.data(), dz.data(), dnum, ix.data(), 
               iy.data(), h_iz.data(), inum, area, avg_dist.data());
  } else {
    printf("Verification disabled\n");
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  size_t dnum_bytes = dnum * sizeof(float);
  size_t inum_bytes = inum * sizeof(float);

  float *d_dx = sycl::malloc_device<float>(dnum, q);
  q.memcpy(d_dx, dx.data(), dnum_bytes);

  float *d_dy = sycl::malloc_device<float>(dnum, q);
  q.memcpy(d_dy, dy.data(), dnum_bytes);

  float *d_dz = sycl::malloc_device<float>(dnum, q);
  q.memcpy(d_dz, dz.data(), dnum_bytes);

  float *d_avg_dist = sycl::malloc_device<float>(dnum, q);
  q.memcpy(d_avg_dist, avg_dist.data(), dnum_bytes);

  float *d_ix = sycl::malloc_device<float>(inum, q);
  q.memcpy(d_ix, ix.data(), inum_bytes);

  float *d_iy = sycl::malloc_device<float>(inum, q);
  q.memcpy(d_iy, iy.data(), inum_bytes);

  float *d_iz = sycl::malloc_device<float>(inum, q);
  q.memcpy(d_iz, iz.data(), inum_bytes);

  sycl::range<1> gws ((inum + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE);
  sycl::range<1> lws (BLOCK_SIZE);

  // Weighted Interpolate using AIDW
  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for<class aidw>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      AIDW_Kernel(d_dx, 
                  d_dy,
                  d_dz,
                  dnum,
                  d_ix,
                  d_iy,
                  d_iz,
                  inum,
                  area,
                  d_avg_dist,
                  item);
    });
  });
  
  q.memcpy(iz.data(), d_iz, inum_bytes).wait();

  if (check) {
    bool ok = verify (iz.data(), h_iz.data(), inum, EPS);
    printf("%s\n", ok ? "PASS" : "FAIL");
    if (!ok) exit(1);
  }

  q.submit([&] (sycl::handler &cgh) {
    sycl::local_accessor<float, 1> sdx(sycl::range<1>(BLOCK_SIZE), cgh);
    sycl::local_accessor<float, 1> sdy(sycl::range<1>(BLOCK_SIZE), cgh);
    sycl::local_accessor<float, 1> sdz(sycl::range<1>(BLOCK_SIZE), cgh);
    cgh.parallel_for<class aidw_tiled>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      AIDW_Kernel_Tiled(d_dx, 
                        d_dy,
                        d_dz,
                        dnum,
                        d_ix,
                        d_iy,
                        d_iz,
                        inum,
                        area,
                        d_avg_dist,
                        sdx.get_multi_ptr<sycl::access::decorated::no>().get(),
                        sdy.get_multi_ptr<sycl::access::decorated::no>().get(),
                        sdz.get_multi_ptr<sycl::access::decorated::no>().get(),
                        item);
    });
  });
  
  q.memcpy(iz.data(), d_iz, inum_bytes).wait();
  if (check) {
    bool ok = verify (iz.data(), h_iz.data(), inum, EPS);
    printf("%s\n", ok ? "PASS" : "FAIL");
    if (!ok) exit(1);
  }

  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < iterations; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        AIDW_Kernel(d_dx, 
                    d_dy,
                    d_dz,
                    dnum,
                    d_ix,
                    d_iy,
                    d_iz,
                    inum,
                    area,
                    d_avg_dist,
                    item);
      });
    });
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of AIDW_Kernel       %f (s)\n", (time * 1e-9f) / iterations);

  start = std::chrono::steady_clock::now();

  for (int i = 0; i < iterations; i++) {
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<float, 1> sdx(sycl::range<1>(BLOCK_SIZE), cgh);
      sycl::local_accessor<float, 1> sdy(sycl::range<1>(BLOCK_SIZE), cgh);
      sycl::local_accessor<float, 1> sdz(sycl::range<1>(BLOCK_SIZE), cgh);
      cgh.parallel_for(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        AIDW_Kernel_Tiled(d_dx, 
                          d_dy,
                          d_dz,
                          dnum,
                          d_ix,
                          d_iy,
                          d_iz,
                          inum,
                          area,
                          d_avg_dist,
                          sdx.get_multi_ptr<sycl::access::decorated::no>().get(),
                          sdy.get_multi_ptr<sycl::access::decorated::no>().get(),
                          sdz.get_multi_ptr<sycl::access::decorated::no>().get(),
                          item);
      });
    });
  }

  q.wait();
  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average execution time of AIDW_Kernel_Tiled %f (s)\n", (time * 1e-9f) / iterations);

  sycl::free(d_dx, q);
  sycl::free(d_dy, q);
  sycl::free(d_dz, q);
  sycl::free(d_ix, q);
  sycl::free(d_iy, q);
  sycl::free(d_iz, q);
  sycl::free(d_avg_dist, q);

  return 0;
}
