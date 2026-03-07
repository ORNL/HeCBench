/*
 * Copyright (c) 2016 University of Cordoba and University of Illinois
 * All rights reserved.
 *
 * Developed by:    IMPACT Research Group
 *                  University of Cordoba and University of Illinois
 *                  http://impact.crhc.illinois.edu/
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the 
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *      > Redistributions of source code must retain the above copyright notice,
 *        this list of conditions and the following disclaimers.
 *      > Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimers in the
 *        documentation and/or other materials provided with the distribution.
 *      > Neither the names of IMPACT Research Group, University of Cordoba, 
 *        University of Illinois nor the names of its contributors may be used 
 *        to endorse or promote products derived from this Software without 
 *        specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 */

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <chrono>
#include <iostream>
#include <sycl/sycl.hpp>
#include <cstdlib>


#if DOUBLE_PRECISION
#define FLOAT double
#else
#define FLOAT float
#endif

typedef struct {
  FLOAT x;
  FLOAT y;
  FLOAT z;
} XYZ;

#define divceil(n, m) (((n)-1) / (m) + 1)

// Params ---------------------------------------------------------------------
struct Params {

  int         work_group_size;
  const char *file_name;
  int         in_size_i;
  int         in_size_j;
  int         out_size_i;
  int         out_size_j;

  Params(int argc, char **argv) {
    work_group_size = 256;
    file_name     = "input/control.txt";
    in_size_i = in_size_j = 3;
    out_size_i = out_size_j = 300;
    int opt;
    while((opt = getopt(argc, argv, "hp:d:i:g:t:w:r:a:f:m:n:")) >= 0) {
      switch(opt) {
        case 'h':
          usage();
          exit(0);
          break;
        case 'g': work_group_size = atoi(optarg); break;
        case 'f': file_name = optarg; break;
        case 'm': in_size_i = in_size_j = atoi(optarg); break;
        case 'n': out_size_i = out_size_j = atoi(optarg); break;
        default:
            fprintf(stderr, "\nUnrecognized option!\n");
            usage();
            exit(0);
      }
    }
  }

  void usage() {
    fprintf(stderr,
        "\nUsage:  ./main [options]"
        "\n"
        "\nGeneral options:"
        "\n    -h        help"
        "\n    -g <G>    # device work-group size (default=256)"
        "\n"
        "\n"
        "\nBenchmark-specific options:"
        "\n    -f <F>    name of input file with control points (default=input/control.txt)"
        "\n    -m <N>    input size in both dimensions (default=3)"
        "\n    -n <R>    output resolution in both dimensions (default=300)"
        "\n");
  }
};

// Input Data -----------------------------------------------------------------
void read_input(XYZ *in, const Params &p) {

  // Open input file
  FILE *f = NULL;
  f       = fopen(p.file_name, "r");
  if(f == NULL) {
    puts("Error opening file");
    exit(-1);
  } else {
    printf("Read data from file %s\n", p.file_name);
  } 


  // Store points from input file to array
  int k = 0, ic = 0;
  XYZ v[10000];
#if DOUBLE_PRECISION
  while(fscanf(f, "%lf,%lf,%lf", &v[ic].x, &v[ic].y, &v[ic].z) == 3)
#else
    while(fscanf(f, "%f,%f,%f", &v[ic].x, &v[ic].y, &v[ic].z) == 3)
#endif
    {
      ic++;
    }
  for(int i = 0; i <= p.in_size_i; i++) {
    for(int j = 0; j <= p.in_size_j; j++) {
      in[i * (p.in_size_j + 1) + j].x = v[k].x;
      in[i * (p.in_size_j + 1) + j].y = v[k].y;
      in[i * (p.in_size_j + 1) + j].z = v[k].z;
      //k++;
      k = (k + 1) % 16;
    }
  }
}

inline int compare_output(XYZ *outp, XYZ *outpCPU, int NI, int NJ, int RESOLUTIONI, int RESOLUTIONJ) {
  double sum_delta2, sum_ref2, L1norm2;
  sum_delta2 = 0;
  sum_ref2   = 0;
  L1norm2    = 0;
  for(int i = 0; i < RESOLUTIONI; i++) {
    for(int j = 0; j < RESOLUTIONJ; j++) {
      sum_delta2 += std::fabs(outp[i * RESOLUTIONJ + j].x - outpCPU[i * RESOLUTIONJ + j].x);
      sum_ref2 += std::fabs(outpCPU[i * RESOLUTIONJ + j].x);
      sum_delta2 += std::fabs(outp[i * RESOLUTIONJ + j].y - outpCPU[i * RESOLUTIONJ + j].y);
      sum_ref2 += std::fabs(outpCPU[i * RESOLUTIONJ + j].y);
      sum_delta2 += std::fabs(outp[i * RESOLUTIONJ + j].z - outpCPU[i * RESOLUTIONJ + j].z);
      sum_ref2 += std::fabs(outpCPU[i * RESOLUTIONJ + j].z);
    }
  }
  L1norm2 = (double)(sum_delta2 / sum_ref2);
  if(L1norm2 >= 1e-6){
    printf("Test failed\n");
    return 1;
  }
  return 0;
}

// BezierBlend (http://paulbourke.net/geometry/bezier/)
inline FLOAT BezierBlend(int k, FLOAT mu, int n) {
  int nn, kn, nkn;
  FLOAT   blend = 1;
  nn        = n;
  kn        = k;
  nkn       = n - k;
  while(nn >= 1) {
    blend *= nn;
    nn--;
    if(kn > 1) {
      blend /= (FLOAT)kn;
      kn--;
    }
    if(nkn > 1) {
      blend /= (FLOAT)nkn;
      nkn--;
    }
  }
  if(k > 0)
    blend *= sycl::pow(mu, (FLOAT)k);
  if(n - k > 0)
    blend *= sycl::pow(1 - mu, (FLOAT)(n - k));
  return (blend);
}

// Sequential implementation for comparison purposes
void BezierCPU(const XYZ *inp, XYZ *outp, const int NI, const int NJ, const int RESOLUTIONI, const int RESOLUTIONJ) {
  int i, j, ki, kj;
  FLOAT   mui, muj, bi, bj;
  for(i = 0; i < RESOLUTIONI; i++) {
    mui = i / (FLOAT)(RESOLUTIONI - 1);
    for(j = 0; j < RESOLUTIONJ; j++) {
      muj     = j / (FLOAT)(RESOLUTIONJ - 1);
      XYZ out = {0, 0, 0};
      for(ki = 0; ki <= NI; ki++) {
        bi = BezierBlend(ki, mui, NI);
        for(kj = 0; kj <= NJ; kj++) {
          bj = BezierBlend(kj, muj, NJ);
          out.x += (inp[ki * (NJ + 1) + kj].x * bi * bj);
          out.y += (inp[ki * (NJ + 1) + kj].y * bi * bj);
          out.z += (inp[ki * (NJ + 1) + kj].z * bi * bj);
        }
      }
      outp[i * RESOLUTIONJ + j] = out;
    }
  }
}

void run(XYZ *in, int in_size_i, int in_size_j, int out_size_i, int out_size_j, const Params &p) {

  XYZ *cpu_out = (XYZ *)malloc(out_size_i * out_size_j * sizeof(XYZ));
  XYZ *gpu_out = (XYZ *)malloc(out_size_i * out_size_j * sizeof(XYZ));

  // CPU run
  auto start = std::chrono::steady_clock::now();
  BezierCPU(in, cpu_out, in_size_i, in_size_j, out_size_i, out_size_j);
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "host execution time: " << time << " ms" << std::endl;

  // Device run
#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  int in_size   = (in_size_i + 1) * (in_size_j + 1);
  int out_size  = out_size_i * out_size_j;

  XYZ *d_in = sycl::malloc_device<XYZ>(in_size, q);
  q.memcpy(d_in, in, sizeof(XYZ) * in_size);

  XYZ *d_out = sycl::malloc_device<XYZ>(out_size, q);

  size_t lws = p.work_group_size;
  size_t gws = (out_size_i + p.work_group_size - 1) / p.work_group_size * p.work_group_size;

  q.wait();
  auto kstart = std::chrono::steady_clock::now();

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<class bs>(
      sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      int i, j, ki, kj;
      FLOAT   mui, muj, bi, bj;

      i = item.get_global_id(0);
      if (i >= out_size_i) return;

      mui = i / (FLOAT)(out_size_i - 1);
      for(j = 0; j < out_size_j; j++) {
        muj     = j / (FLOAT)(out_size_j - 1);
        XYZ out = {0, 0, 0};

        //#pragma unroll
        for(ki = 0; ki <= in_size_i; ki++) {
          bi = BezierBlend(ki, mui, in_size_i);
          //#pragma unroll
          for(kj = 0; kj <= in_size_j; kj++) {
            bj = BezierBlend(kj, muj, in_size_j);
            out.x += (d_in[ki * (in_size_j + 1) + kj].x * bi * bj);
            out.y += (d_in[ki * (in_size_j + 1) + kj].y * bi * bj);
            out.z += (d_in[ki * (in_size_j + 1) + kj].z * bi * bj);
          }
        }
        d_out[i * out_size_j + j] = out;
      }
    });
  });

  q.wait();
  auto kend = std::chrono::steady_clock::now();
  auto ktime = std::chrono::duration_cast<std::chrono::milliseconds>(kend - kstart).count();
  std::cout << "kernel execution time: " << ktime << " ms" << std::endl;

  q.memcpy(gpu_out, d_out, sizeof(XYZ) * out_size).wait();

  // Verify
  int status = compare_output(gpu_out, cpu_out, in_size_i, in_size_j, out_size_i, out_size_j);
  printf("%s\n", (status == 0) ? "PASS" : "FAIL");

  free(cpu_out);
  free(gpu_out);
  sycl::free(d_in, q);
  sycl::free(d_out, q);
}

int main(int argc, char **argv) {

  const Params p(argc, argv);
  int in_size   = (p.in_size_i + 1) * (p.in_size_j + 1) * sizeof(XYZ);
  //int out_size  = p.out_size_i * p.out_size_j * sizeof(XYZ);

  // load data into h_in
  XYZ* h_in = (XYZ *)malloc(in_size);
  read_input(h_in, p);

  // run the app on the cpu and gpu
  run(h_in, p.in_size_i, p.in_size_j, p.out_size_i, p.out_size_j, p);

  free(h_in);
  if (!(status == 0)) return 1;
  return 0;
}
