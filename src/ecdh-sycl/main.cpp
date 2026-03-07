#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <math.h>
#include <sycl/sycl.hpp>
#include "ecdh.h"

#define P_x 5
#define P_y 1
#define MODULUS 17
#define a 2

int main(int argc, char **argv)
{
  if (argc != 3) {
    printf("Usage: %s <positive number of keys> <repeat>\n", argv[0]);
    return 1;
  }
  const int num_pk = atoi(argv[1]); 
  const int repeat = atoi(argv[2]); 

  // timing fast and slow versions for making public key
  struct timeval start_fast, end_fast, start_slow, end_slow;

  const int pk_x_size = num_pk * sizeof(int);
  const int pk_y_size = num_pk * sizeof(int);

  int *pk_slow_x = (int*) malloc (pk_x_size);
  int *pk_slow_y = (int*) malloc (pk_y_size);
  int *pk_fast_x = (int*) malloc (pk_x_size);
  int *pk_fast_y = (int*) malloc (pk_y_size);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  int *d_pk_x = sycl::malloc_device<int>(num_pk, q);
  int *d_pk_y = sycl::malloc_device<int>(num_pk, q);
  
  sycl::range<1> gws ((num_pk + 255) / 256 * 256);
  sycl::range<1> lws (256);

  gettimeofday(&start_slow,NULL);

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class ecdh_slow>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        k_slow(item, 18, P_x, P_y,
               d_pk_x, d_pk_y,
               MODULUS, a, num_pk);  
      });
    });
  }

  q.wait();
  gettimeofday(&end_slow,NULL);
  double elapsed_slow = (((end_slow.tv_sec*1000000.0 + end_slow.tv_usec) -
                        (start_slow.tv_sec*1000000.0 + start_slow.tv_usec)) / 1000000.00);

  printf("Average time (slow kernel): %f s\n", elapsed_slow / repeat);

  q.memcpy(pk_slow_x, d_pk_x, pk_x_size);
  q.memcpy(pk_slow_y, d_pk_y, pk_y_size);

  q.wait();

  gettimeofday(&start_fast,NULL);

  for (int i = 0; i < repeat; i++) {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class ecdh_fast>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        k_fast(item, 18, P_x, P_y,
               d_pk_x, d_pk_y,
               MODULUS, a, num_pk);  
      });
    });
  }

  q.wait();
  gettimeofday(&end_fast,NULL);
  double elapsed_fast = (((end_fast.tv_sec*1000000.0 + end_fast.tv_usec) -
                        (start_fast.tv_sec*1000000.0 + start_fast.tv_usec)) / 1000000.00);

  printf("Average time (fast kernel): %f s\n", elapsed_fast / repeat);

  q.memcpy(pk_fast_x, d_pk_x, pk_x_size);
  q.memcpy(pk_fast_y, d_pk_y, pk_y_size);

  q.wait();
  
  bool fail_pk_x = memcmp(pk_slow_x, pk_fast_x, pk_x_size);
  bool fail_pk_y = memcmp(pk_slow_y, pk_fast_y, pk_x_size);
  printf("%s\n", (fail_pk_x || fail_pk_y) ? "FAIL" : "PASS");

  sycl::free(d_pk_x, q);
  sycl::free(d_pk_y, q);
  free(pk_slow_x);
  free(pk_slow_y);
  free(pk_fast_x);
  free(pk_fast_y);
  if ((fail_pk_x || fail_pk_y)) return 1;
  return 0;
}
