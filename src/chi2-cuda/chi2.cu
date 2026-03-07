#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <chrono>
#include <random>
#include <cuda.h>
#include "reference.h"

__global__ void chi_kernel(
  const unsigned int rows,
  const unsigned int cols,
  const int cRows,
  const int contRows,
  const unsigned char *__restrict__ snpdata,
  float *__restrict__ results)
{
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
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
  for ( m = 0 ; m < cRows ; m++ ) {
    y = snpdata[(size_t)m * (size_t)cols + tid];
    if ( y == '0') { cases[0]++; }
    else if ( y == '1') { cases[1]++; }
    else if ( y == '2') { cases[2]++; }
  }

  // read controls: each thread reads a column of snpdata matrix
  for ( n = cRows ; n < cRows + contRows ; n++ ) {
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
  results[tid] = chisquare;
}

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

  // allocate SNP device data
  unsigned char *d_data;
  float *d_results;
  cudaMalloc((void**) &d_data, snpdata_size);
  cudaMalloc((void**) &d_results, result_size);

  cudaMemcpy(d_data, dataT, snpdata_size, cudaMemcpyHostToDevice);

  unsigned jobs = cols;
  int nblocks = (jobs + nthreads - 1)/nthreads;

  cudaDeviceSynchronize();
  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < repeat; i++) {
    chi_kernel <<< dim3(nblocks), dim3(nthreads) >>> (rows,cols,ncases,ncontrols,d_data,d_results);
  }

  cudaDeviceSynchronize();
  auto end = std::chrono::high_resolution_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average chi_kernel execution time = %f (s)\n", time * 1e-9f / repeat);

  cudaMemcpy(h_results, d_results, result_size, cudaMemcpyDeviceToHost);

  cudaFree(d_data);
  cudaFree(d_results);

  start = std::chrono::high_resolution_clock::now();

  cpu_kernel(rows,cols,ncases,ncontrols,dataT,cpu_results);

  end = std::chrono::high_resolution_clock::now();
  time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Host execution time = %f (s)\n", time * 1e-9f);

  // verify
  int error = 0;
  for(unsigned int k = 0; k < jobs; k++) {
    if (fabs(cpu_results[k] - h_results[k]) > 1e-4) error++;
  }
  printf("%s\n", error ? "FAIL" : "PASS");

  free(dataT);
  free(h_results);
  free(cpu_results);

  if (error) return 1;
  return 0;
}
