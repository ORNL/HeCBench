#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <chrono>
#include <random>
#include <sycl/sycl.hpp>
#include "reference.h"

int main(int argc, char* argv[]) {

  if (argc != 7) {
    printf("Usage: %s <rows> <cols> <cases> <controls> "
           "<threads> <repeat>\n", argv[0]);
    return 1;
  }

  unsigned int rows = atoi(argv[1]);
  unsigned int cols = atoi(argv[2]);
  int ncases = atoi(argv[3]);
  int ncontrols = atoi(argv[4]);
  int nthreads = atoi(argv[5]);
  int repeat = atoi(argv[6]);

  printf("Individuals=%d SNPs=%d cases=%d controls=%d nthreads=%d\n",
         rows,cols,ncases,ncontrols,nthreads);

  size_t size = (size_t)rows * (size_t)cols;
  printf("Size of the data = %lu\n",size);

  // allocate SNP host data 
  size_t snpdata_size = sizeof(unsigned char) * size;
  size_t result_size = sizeof(float) * cols;

  unsigned char *dataT = (unsigned char*)malloc(snpdata_size);
  float* h_results = (float*) malloc(result_size);
  float* cpu_results = (float*) malloc(result_size);

  if(dataT == NULL || h_results == NULL || cpu_results == NULL) {
    printf("ERROR: Memory for data not allocated.\n");
    if (dataT) free(dataT);
    if (h_results) free(h_results);
    return 1;
  }

  std::mt19937 gen(19937); // mersenne_twister_engin
  std::uniform_int_distribution<> distrib(0, 2);
  for (size_t i = 0; i < snpdata_size; i++) {
    dataT[i] = distrib(gen) + '0';
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // allocate SNP device data
  unsigned char *snpdata = sycl::malloc_device<unsigned char>(size, q);
  float *chi_result = sycl::malloc_device<float>(cols, q);

  q.memcpy(snpdata, dataT, snpdata_size);

  unsigned jobs = cols;
  sycl::range<1> gws ((jobs + nthreads - 1)/nthreads * nthreads);
  sycl::range<1> lws (nthreads);

  q.wait();
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<class chi2>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {

        int tid  = item.get_global_id(0);
        if (tid >= cols) return;

        unsigned char y;
        int m, n;
        unsigned int p = 0;
        int tot_cases = 1;
        int tot_controls= 1;
        int total = 1;
        float chisquare = 0.0f;
        float exp[3];        
        float Conexpected[3];        
        float Cexpected[3];
        float numerator1;
        float numerator2;

        int cases[3] = {1,1,1};
        int controls[3] = {1,1,1};

        // read cases: each thread reads a column of snpdata matrix
        for ( m = 0 ; m < ncases ; m++ ) {
          y = snpdata[(size_t)m * (size_t)cols + tid];
          if ( y == '0') { cases[0]++; }
          else if ( y == '1') { cases[1]++; }
          else if ( y == '2') { cases[2]++; }
        }

        // read controls: each thread reads a column of snpdata matrix
        for ( n = ncases ; n < ncases + ncontrols ; n++ ) {
          y = snpdata[(size_t)n * (size_t)cols + tid];
          if ( y == '0' ) { controls[0]++; }
          else if ( y == '1') { controls[1]++; }
          else if ( y == '2') { controls[2]++; }
        }

        for( p = 0 ; p < 3; p++ ) {
          tot_cases += cases[p];
          tot_controls += controls[p];
        }
        total = tot_cases + tot_controls;

        for( p = 0 ; p < 3; p++ ) {
          exp[p] = (float)cases[p] + controls[p]; 
          Cexpected[p] = tot_cases * exp[p] / total;
          Conexpected[p] = tot_controls * exp[p] / total;
          numerator1 = (float)cases[p] - Cexpected[p];
          numerator2 = (float)controls[p] - Conexpected[p];
          chisquare += numerator1 * numerator1 / Cexpected[p] +
                       numerator2 * numerator2 / Conexpected[p];
        }
        chi_result[tid] = chisquare;
      });
    });
  }
  q.wait();

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time = %f (s)\n", time * 1e-9f / repeat);

  q.memcpy(h_results, chi_result, result_size).wait();

  sycl::free(snpdata, q);
  sycl::free(chi_result, q);

  start = std::chrono::steady_clock::now();

  cpu_kernel(rows,cols,ncases,ncontrols,dataT,cpu_results);

  end = std::chrono::steady_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Host execution time = %f (s)\n", time * 1e-9f);

  // verify
  int error = 0;
  for(unsigned int k = 0; k < jobs; k++) {
    if (std::fabs(cpu_results[k] - h_results[k]) > 1e-4) error++;
  }
  printf("%s\n", error ? "FAIL" : "PASS");
  if (error) exit(1);

  free(dataT);
  free(h_results);
  free(cpu_results);

  return 0;
}
