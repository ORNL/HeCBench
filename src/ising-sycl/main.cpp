/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <chrono>
#include <fstream>
#include <getopt.h>
#include <iostream>
#include <string>
#include <cstring>

#ifdef MKLRAND
#include <mkl_rng_sycl.hpp>
#endif
#include <sycl/sycl.hpp>

#include "reference.h"
#include <cstdlib>

#define TCRIT    2.26918531421f
#define THREADS  128


// Initialize lattice spins
void init_spins(signed char *lattice,
                const float *randvals,
                const long long nx,
                const long long ny,
                sycl::nd_item<1> &item)
{
  const long long tid = item.get_global_id(0);
  if (tid >= nx * ny) return;

  float randval = randvals[tid];
  signed char val = (randval < 0.5f) ? -1 : 1;
  lattice[tid] = val;
}

template<bool is_black>
void update_lattice(      signed char *lattice,
                    const signed char *op_lattice,
                    const float *randvals,
                    const float inv_temp,
                    const long long nx,
                    const long long ny,
                    sycl::nd_item<1> &item)
{
  const long long tid = item.get_global_id(0);
  const int i = tid / ny;
  const int j = tid % ny;

  if (i >= nx || j >= ny) return;

  // Set stencil indices with periodicity
  int ipp = (i + 1 < nx) ? i + 1 : 0;
  int inn = (i - 1 >= 0) ? i - 1: nx - 1;
  int jpp = (j + 1 < ny) ? j + 1 : 0;
  int jnn = (j - 1 >= 0) ? j - 1: ny - 1;

  // Select off-column index based on color and row index parity
  int joff;
  if (is_black) {
    joff = (i % 2) ? jpp : jnn;
  } else {
    joff = (i % 2) ? jnn : jpp;
  }

  // Compute sum of nearest neighbor spins
  signed char nn_sum = op_lattice[inn * ny + j] + op_lattice[i * ny + j] +
                       op_lattice[ipp * ny + j] + op_lattice[i * ny + joff];

  // Determine whether to flip spin
  signed char lij = lattice[i * ny + j];
  float acceptance_ratio = sycl::exp(-2.0f * inv_temp * nn_sum * lij);
  if (randvals[i*ny + j] < acceptance_ratio) {
    lattice[i * ny + j] = -lij;
  }
}


void update(sycl::queue &q,
            signed char *d_lattice_b, 
            signed char *d_lattice_w, 
            float *d_randvals, 
#ifdef MKLRAND
            mkl::rng::philox4x32x10 &rng,
            mkl::rng::uniform<float> &distr,
#endif
	    const float inv_temp, const long long nx, const long long ny) {


  int blocks = (nx * ny/2 + THREADS - 1) / THREADS * THREADS;

  // Update black
#ifdef MKLRAND
  mkl::rng::generate(distr, rng, nx * ny / 2, d_randvals);
#endif
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class update_black>(
      sycl::nd_range<1>(sycl::range<1>(blocks), sycl::range<1>(THREADS)),
      [=](sycl::nd_item<1> item) {
      update_lattice<true>(d_lattice_b, d_lattice_w, d_randvals,
                           inv_temp, nx, ny / 2, item);
    });
  });

  // Update white
#ifdef MKLRAND
  mkl::rng::generate(distr, rng, nx * ny / 2, d_randvals);
#endif
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class update_white>(
      sycl::nd_range<1>(sycl::range<1>(blocks), sycl::range<1>(THREADS)),
      [=](sycl::nd_item<1> item) {
      update_lattice<false>(d_lattice_w, d_lattice_b, d_randvals,
                            inv_temp, nx, ny / 2, item);
    });
  });
}

static void usage(const char *pname) {

  const char *bname = rindex(pname, '/');
  if (!bname) {bname = pname;}
  else        {bname++;}

  fprintf(stdout,
          "Usage: %s [options]\n"
          "options:\n"
          "\t-x|--lattice-n <LATTICE_N>\n"
          "\t\tnumber of lattice rows\n"
          "\n"
          "\t-y|--lattice_m <LATTICE_M>\n"
          "\t\tnumber of lattice columns\n"
          "\n"
          "\t-w|--nwarmup <NWARMUP>\n"
          "\t\tnumber of warmup iterations\n"
          "\n"
          "\t-n|--niters <NITERS>\n"
          "\t\tnumber of trial iterations\n"
          "\n"
          "\t-a|--alpha <ALPHA>\n"
          "\t\tcoefficient of critical temperature\n"
          "\n"
          "\t-s|--seed <SEED>\n"
          "\t\tseed for random number generation\n\n",
          bname);
  exit(EXIT_SUCCESS);
}

int main(int argc, char **argv) {

  // Defaults
  long long nx = 5120;
  long long ny = 5120;
  float alpha = 0.1f;
  int nwarmup = 100;
  int niters = 1000;
  unsigned long long seed = 1234ULL;

  while (1) {
    static struct option long_options[] = {
        {     "lattice-n", required_argument, 0, 'x'},
        {     "lattice-m", required_argument, 0, 'y'},
        {         "alpha", required_argument, 0, 'y'},
        {          "seed", required_argument, 0, 's'},
        {       "nwarmup", required_argument, 0, 'w'},
        {        "niters", required_argument, 0, 'n'},
        {          "help",       no_argument, 0, 'h'},
        {               0,                 0, 0,   0}
    };

    int option_index = 0;
    int ch = getopt_long(argc, argv, "x:y:a:s:w:n:h", long_options, &option_index);
    if (ch == -1) break;

    switch(ch) {
      case 0:
        break;
      case 'x':
        nx = atoll(optarg); break;
      case 'y':
        ny = atoll(optarg); break;
      case 'a':
        alpha = atof(optarg); break;
      case 's':
        seed = atoll(optarg); break;
      case 'w':
        nwarmup = atoi(optarg); break;
      case 'n':
        niters = atoi(optarg); break;
      case 'h':
        usage(argv[0]); break;
      case '?':
        exit(EXIT_FAILURE);
      default:
        fprintf(stderr, "unknown option: %c\n", ch);
        exit(EXIT_FAILURE);
    }
  }

  // Check arguments
  if (nx % 2 != 0 || ny % 2 != 0) {
    fprintf(stderr, "ERROR: Lattice dimensions must be even values.\n");
    exit(EXIT_FAILURE);
  }

  float inv_temp = 1.0f / (alpha*TCRIT);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

#ifdef MKLRAND
  // Setup MKL random number generator
  mkl::rng::philox4x32x10 rng(q, seed);
  mkl::rng::uniform<float> distr;
#else
  // for verification across difference platforms, generate random values once
  srand(seed);
  float* randvals_host = (float*) malloc(nx * ny/2 * sizeof(float));
  for (int i = 0; i < nx * ny/2; i++)
    randvals_host[i] = (float)rand() / (float)RAND_MAX;
#endif

  signed char *lattice_b_r, *lattice_w_r;
  lattice_b_r = (signed char*) malloc(nx * ny/2 * sizeof(signed char));
  lattice_w_r = (signed char*) malloc(nx * ny/2 * sizeof(signed char));

  float *d_randvals = sycl::malloc_device<float>(nx * ny / 2, q);

#ifndef MKLRAND
  q.memcpy(d_randvals, randvals_host, nx*ny/2*sizeof(float));
#endif

  signed char *d_lattice_b = sycl::malloc_device<signed char>(nx * ny / 2, q);
  signed char *d_lattice_w = sycl::malloc_device<signed char>(nx * ny / 2, q);

  int blocks = (nx * ny/2 + THREADS - 1) / THREADS * THREADS ;

  sycl::range<1> gws (blocks);
  sycl::range<1> lws (THREADS);

#ifdef MKLRAND
  mkl::rng::generate(distr, rng, nx * ny / 2, d_randvals);
#endif

  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class init_lattice_black>(
      sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
      init_spins(d_lattice_b, d_randvals, nx, ny / 2, item);
    });
  });

#ifdef MKLRAND
  mkl::rng::generate(distr, rng, nx * ny / 2, d_randvals);
#endif
  q.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class init_lattice_white>(
      sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
      init_spins(d_lattice_w, d_randvals, nx, ny / 2, item);
    });
  });

  // Warmup iterations
  printf("Starting warmup...\n");
  for (int i = 0; i < nwarmup; i++) {
    update(q, d_lattice_b, d_lattice_w, d_randvals, 
#ifdef MKLRAND
        rng, 
        distr, 
#endif
        inv_temp, nx, ny);
  }

  q.wait();

  printf("Starting trial iterations...\n");
  auto t0 = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < niters; i++) {
    update(q, d_lattice_b, d_lattice_w, d_randvals, 
#ifdef MKLRAND
        rng, 
        distr, 
#endif
        inv_temp, nx, ny);
  }

  q.wait();
  auto t1 = std::chrono::high_resolution_clock::now();

  double duration = (double) std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
  printf("REPORT:\n");
  printf("\tnGPUs: %d\n", 1);
  printf("\ttemperature: %f * %f\n", alpha, TCRIT);
  printf("\tseed: %llu\n", seed);
  printf("\twarmup iterations: %d\n", nwarmup);
  printf("\ttrial iterations: %d\n", niters);
  printf("\tlattice dimensions: %lld x %lld\n", nx, ny);
  printf("\telapsed time: %f sec\n", duration * 1e-6);
  printf("\tupdates per ns: %f\n", (double) (nx * ny) * niters / duration * 1e-3);

  signed char* lattice_b_h = (signed char*) malloc(nx * ny/2 * sizeof(*lattice_b_h));
  signed char* lattice_w_h = (signed char*) malloc(nx * ny/2 * sizeof(*lattice_w_h));

  q.memcpy(lattice_b_h, d_lattice_b, nx * ny/2 * sizeof(signed char));
  q.memcpy(lattice_w_h, d_lattice_w, nx * ny/2 * sizeof(signed char));

  q.wait();

  printf("Starting verification iterations ...\n");
  init_spins_ref(lattice_b_r, randvals_host, nx, ny/2);
  init_spins_ref(lattice_w_r, randvals_host, nx, ny/2);
  for (int i = 0; i < nwarmup + niters; i++) {
    update_ref(lattice_b_r, lattice_w_r, randvals_host, inv_temp, nx, ny);
  }

  bool ok = true;
  for (int i = 0; i < nx*ny/2; i++) {
    ok  = (lattice_b_h[i] == lattice_b_r[i]) && 
          (lattice_w_h[i] == lattice_w_r[i]);
    if (!ok) break;
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

#ifndef MKLRAND
  free(randvals_host);
#endif
  free(lattice_b_h);
  free(lattice_w_h);
  free(lattice_b_r);
  free(lattice_w_r);

  sycl::free(d_lattice_b, q);
  sycl::free(d_lattice_w, q);
  sycl::free(d_randvals, q);

  return 0;
}
