//
// Program to solve Laplace equation on a regular 3D grid
//

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <chrono>
#include <sycl/sycl.hpp>
#include "kernel.h"
#include "reference.h"

// declaration, forward

void printHelp(void);

// Main program

int main(int argc, char **argv){

  // check command line inputs
  if(argc != 6) {
    printHelp();
    return 1;
  }

  const int NX = atoi(argv[1]);
  const int NY = atoi(argv[2]);
  const int NZ = atoi(argv[3]);
  const int REPEAT = atoi(argv[4]);
  const int verify = atoi(argv[5]);

  // NX is a multiple of 32
  if (NX <= 0 || NX % 32 != 0 || NY <= 0 || NZ <= 0 || REPEAT <= 0) return 1;

  printf("\nGrid dimensions: %d x %d x %d\n", NX, NY, NZ);
  printf("Result verification %s \n", verify ? "enabled" : "disabled");
 
  // allocate memory for arrays

  const size_t grid3D_size = NX * NY * NZ ;
  const size_t grid3D_bytes = grid3D_size * sizeof(float);

  float *h_u1 = (float *) malloc (grid3D_bytes);
  float *h_u2 = (float *) malloc (grid3D_bytes);
  float *h_u3 = (float *) malloc (grid3D_bytes);

  const int pitch = NX;

  // initialise u1
  int i, j, k;
    
  for (k=0; k<NZ; k++) {
    for (j=0; j<NY; j++) {
      for (i=0; i<NX; i++) {
        int ind = i + j*NX + k*NX*NY;
        if (i==0 || i==NX-1 || j==0 || j==NY-1|| k==0 || k==NZ-1)
          h_u1[ind] = 1.0f;           // Dirichlet b.c.'s
        else
          h_u1[ind] = 0.0f;
      }
    }
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  float *d_u1 = sycl::malloc_device<float>(grid3D_size, q);
  q.memcpy(d_u1, h_u1, grid3D_bytes);

  float *d_u2 = sycl::malloc_device<float>(grid3D_size, q);

  // Set up the execution configuration
  const int bx = 1 + (NX-1)/BLOCK_X;
  const int by = 1 + (NY-1)/BLOCK_Y;

  sycl::range<2> lws (BLOCK_Y, BLOCK_X);
  sycl::range<2> gws (by * BLOCK_Y, bx * BLOCK_X);

  printf("\nglobal work size  = %d %d %d \n", bx * BLOCK_X, by * BLOCK_Y, 1);
  printf("local work size = %d %d %d \n", BLOCK_X, BLOCK_Y, 1);

  // Warmup
  q.submit([&] (sycl::handler &cgh) {
    sycl::local_accessor<float, 1> sm (3*KOFF, cgh);
    cgh.parallel_for<class warmup>(
      sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
      laplace3d(item, NX, NY, NZ, pitch,
                d_u1, d_u2,
                sm.get_multi_ptr<sycl::access::decorated::no>().get());
    });
  }).wait();

  // Execute GPU kernel
  auto start = std::chrono::steady_clock::now();

  for (i = 1; i <= REPEAT; ++i) {
    q.submit([&] (sycl::handler &cgh) {
      sycl::local_accessor<float, 1> sm (3*KOFF, cgh);
      cgh.parallel_for<class eval>(
        sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
        laplace3d(item, NX, NY, NZ, pitch,
                  d_u1, d_u2,
                  sm.get_multi_ptr<sycl::access::decorated::no>().get());
      });
    });
    std::swap(d_u1, d_u2);
  }

  q.wait();
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time: %f (s)\n", (time * 1e-9f) / REPEAT);

  q.memcpy(h_u2, d_u1, grid3D_bytes).wait();

  if (verify) {
    // Reference
    for (i = 1; i <= REPEAT; ++i) {
      reference(NX, NY, NZ, h_u1, h_u3);
      std::swap(h_u1, h_u3);
    }

    // verify (may take long for large grid sizes)
    bool ok = true;
    float err = 0.f;
    for (k=0; k<NZ; k++) {
      for (j=0; j<NY; j++) {
        for (i=0; i<NX; i++) {
          int ind = i + j*NX + k*NX*NY;
          err += (h_u1[ind]-h_u2[ind])*(h_u1[ind]-h_u2[ind]);
          if (fabsf(h_u1[ind] - h_u2[ind]) > 1e-3f) {
            ok = false;
          }
        }
      }
    }
    printf("\n RMS error = %f \n", sqrtf(err/ NX*NY*NZ));
    printf(" %s\n", ok ? "PASS" : "FAIL");
  }

 // Release GPU and CPU memory
  sycl::free(d_u1, q);
  sycl::free(d_u2, q);
  free(h_u1);
  free(h_u2);
  free(h_u3);

  if (!ok) return 1;
  return 0;
}


//Print help screen
void printHelp(void)
{
  printf("Usage:  laplace3d [OPTION]...\n");
  printf("6-point stencil 3D Laplace test \n");
  printf("\n");
  printf("Example: run 100 iterations on a 256x128x128 grid\n");
  printf("./main 256 128 128 100 1\n");

  printf("\n");
  printf("Options:\n");
  printf("Grid width\n");
  printf("Grid height\n");
  printf("Grid depth\n");
  printf("Number of repetitions\n");
  printf("verify the result\n");
}
