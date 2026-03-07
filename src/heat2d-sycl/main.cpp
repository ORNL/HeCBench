#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <sys/time.h>
#include <math.h>
#include <utility>
#include <sycl/sycl.hpp>

#define NTX 16
#define NTY 16

double stop_watch(double t0)
{
  double time;
  struct timeval t;
  gettimeofday(&t, NULL);
  time = t.tv_sec * 1e6 + t.tv_usec;
  return time-t0;
}

void usage(char *argv[]) {
  fprintf(stderr, " Usage: %s LX LY NITER\n", argv[0]);
  return;
}

void reference(float *out, const float *in, const float delta, const float norm, const int Lx, const int Ly)
{
  #pragma omp parallel for collapse(2)
  for (int y = 0; y < Ly; y++) {
    for (int x = 0; x < Lx; x++) {
      int v00 = y*Lx + x;
      int v0p = y*Lx + (x + 1)%Lx;
      int v0m = y*Lx + (Lx + x - 1)%Lx;
      int vp0 = ((y+1)%Ly)*Lx + x;
      int vm0 = ((Ly+y-1)%Ly)*Lx + x;
      out[v00] = norm*in[v00] + delta*(in[v0p] + in[v0m] + in[vp0] + in[vm0]);
    }
  }
}

/*
 * Naive implementation of a single iteration of the lapl
 * equation. Each thread takes one site of the output array
 */
void
dev_lapl_iter(float *out, const float *in, const float delta, const float norm, const int Lx, const int Ly,
              const sycl::nd_item<1> &item)
{
  int i = item.get_global_id(0);
  int x = i % Lx;
  int y = i / Lx;
  int v00 = y*Lx + x;
  int v0p = y*Lx + (x + 1)%Lx;
  int v0m = y*Lx + (Lx + x - 1)%Lx;
  int vp0 = ((y+1)%Ly)*Lx + x;
  int vm0 = ((Ly+y-1)%Ly)*Lx + x;
  out[v00] = norm*in[v00] + delta*(in[v0p] + in[v0m] + in[vp0] + in[vm0]);
}

int main(int argc, char *argv[]) {
  /* Check the number of command line arguments */
  if(argc != 4) {
    usage(argv);
    exit(1);
  }
  /* The length of the array in x and y is read from the command
     line */
  int Lx = atoi(argv[1]);
  int Ly = atoi(argv[2]);
  if (Lx % NTX != 0 ||  Ly % NTY != 0) {
    printf("Array length LX and LY must be a multiple of block size %d and %d, respectively\n",
          NTX, NTY);
    exit(1);
  }
  /* The number of iterations */
  int niter = atoi(argv[3]);
  assert(niter >= 1);

  /* Fixed "sigma" */
  const float sigma = 0.01;
  const float xdelta = sigma / (1.0+4.0*sigma);
  const float xnorm = 1.0/(1.0+4.0*sigma);

  printf(" Ly,Lx = %d,%d\n", Ly, Lx);
  printf(" niter = %d\n", niter);

  /* Allocate the buffer for the data */
  srand(123);
  float *buffer = (float*) malloc(sizeof(float)*Lx*Ly);
  float *h_out = (float*) malloc(sizeof(float)*Lx*Ly);
  float *h_in  = (float*) malloc(sizeof(float)*Lx*Ly);
  float *d_res = (float*) malloc(sizeof(float)*Lx*Ly);

  int i, j, x, y;
  for (i = 0; i < Lx; i += 16) {
    x = rand() % Lx;
    for (j = 0; j < Ly; j++)
       buffer[x + j*Lx] = 1.f;
  }
  for (i = 0; i < Ly; i += 16) {
    y = rand() % Ly;
    for (j = 0; j < Lx; j++)
       buffer[j + y*Lx] = 1.f;
  }
  memcpy(h_in, buffer, sizeof(float)*Lx*Ly);

  for(int i=0; i<niter; i++) {
    reference (h_out, h_in, xdelta, xnorm, Lx, Ly);
    std::swap(h_out, h_in);
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  /* Initialize: allocate GPU arrays and load array to GPU */
  float *d_in, *d_out;	/* GPU arrays */
  d_in = sycl::malloc_device<float>(Lx * Ly, q);
  d_out = sycl::malloc_device<float>(Lx * Ly, q);
  q.memcpy(d_in, buffer, sizeof(float) * Lx * Ly);

  sycl::range<1> gws (Ly*Lx);
  sycl::range<1> lws (NTY*NTX);

  /* Do iterations on GPU, record time */
  q.wait();
  double t0 = stop_watch(0);

  /* Fixed number of threads per block (in x- and y-direction), number
     of blocks per direction determined by dimensions Lx, Ly */
  for(i=0; i<niter; i++) {
    q.parallel_for(
      sycl::nd_range<1>(gws, lws), [=](sycl::nd_item<1> item) {
        dev_lapl_iter(d_out, d_in, xdelta, xnorm, Lx, Ly, item);
    });
    std::swap(d_out, d_in);
  }

  q.wait();
  t0 = stop_watch(t0)/(double)niter;

  printf("Device: iters = %8d, (Lx,Ly) = %6d, %6d, t = %8.1f usec/iter, BW = %6.3f GB/s, P = %6.3f Gflop/s\n",
  	 niter, Lx, Ly, t0,
  	 Lx*Ly*sizeof(float)*2.0/(t0*1.0e3),
  	 (Lx*Ly*6.0)/(t0*1.0e3));

  /* copy GPU array to main memory and free GPU arrays */
  q.memcpy(d_res, d_in, sizeof(float) * Lx * Ly).wait();

  // verification
  bool ok = true;
  for (i = 0; i < Lx*Ly; i++) {
    // choose 1e-2 because the error rate increases with the iteration from 1 to 100000
    if ( fabs(h_in[i] - d_res[i]) > 1e-2 ) {
      printf("Mismatch at %d cpu=%f gpu=%f\n", i, h_in[i], d_res[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");

  sycl::free(d_in, q);
  sycl::free(d_out, q);
  free(buffer);
  free(h_out);
  free(h_in);
  free(d_res);
  if (!ok) return 1;
  return 0;
}
