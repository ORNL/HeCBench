/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2014, Erick Lavoie, Faiz Khan, Sujay Kathrotia, Vincent
 * Foley-Bourgon, Laurie Hendren
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 *of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <getopt.h>
#include <chrono>
#include <hip/hip_runtime.h>
#include "reference.h"

#define D_FACTOR (0.85f)
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif

// default values 
const int max_iter = 1000;
const float threshold= 1e-16f;

__global__
void map(const int *__restrict__ pages,
         const float *__restrict__ page_ranks,
               float *__restrict__ maps,
         const unsigned int *__restrict__ noutlinks,
         const int n)
{
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j;
  if(i < n){
    float outbound_rank = page_ranks[i]/(float)noutlinks[i];
    for(j=0; j<n; ++j){
      maps[(size_t)i*n+j] = pages[(size_t)i*n+j]*outbound_rank;
    }
  }
}

__global__
void reduce(      float *__restrict__ page_ranks,
            const float *__restrict__ maps,
            const int n,
                  float *__restrict__ dif)
{

  int j = threadIdx.x + blockIdx.x * blockDim.x;
  int i;
  float new_rank;
  float old_rank;

  if(j<n){
    old_rank = page_ranks[j];
    new_rank = 0.0f;
    for(i=0; i< n; ++i){
      new_rank += maps[(size_t)i*n + j];
    }

    new_rank = ((1.f-D_FACTOR)/n)+(D_FACTOR*new_rank);
    dif[j] = fmaxf(fabsf(new_rank - old_rank), dif[j]);
    page_ranks[j] = new_rank;
  }
}

// generates an array of random pages and their links
int *random_pages(int n, unsigned int *noutlinks, int divisor){
  int i, j, k;
  int *pages = (int*) malloc((size_t)n * n * sizeof(int)); // matrix 1 means link from j->i

  if (divisor <= 0) {
    fprintf(stderr, "ERROR: Invalid divisor '%d' for random initialization, divisor should be greater or equal to 1\n", divisor);
    exit(1);
  }

  for(i=0; i<n; ++i){
    noutlinks[i] = 0;
    for(j=0; j<n; ++j){
      if(i!=j && (abs(rand())%divisor == 0)){
        pages[(size_t)i*n+j] = 1;
        noutlinks[i] += 1;
      }
    }

    // the case with no outlinks is avoided
    if(noutlinks[i] == 0){
      do { k = abs(rand()) % n; } while ( k == i);
      pages[(size_t)i*n + k] = 1;
      noutlinks[i] = 1;
    }
  }
  return pages;
}

void init_array(float *a, int n, float val){
  int i;
  for(i=0; i<n; ++i){
    a[i] = val;
  }
}

void usage(char *argv[]){
  fprintf(stderr, "Usage: %s [-n number of pages] [-i max iterations]"
                  " [-t threshold] [-q divisor for zero density]\n", argv[0]);
}

static struct option size_opts[] =
{
  /* name, has_tag, flag, val*/
  {"number of pages", 1, NULL, 'n'},
  {"max number of iterations", 1, NULL, 'i'},
  {"minimum threshold", 1, NULL, 't'},
  {"divisor for zero density", 1, NULL, 'q'},
  { 0, 0, 0}
};

float maximum_dif(float *difs, int n){
  int i;
  float max = 0.0f;
  for(i=0; i<n; ++i){
    max = difs[i] > max ? difs[i] : max;
  }
  return max;
}

int main(int argc, char *argv[]) {
  int *pages;
  float *maps;
  float *page_ranks;
  unsigned int *noutlinks;
  int t;
  float max_diff, max_diff_ref;

  int i = 0;
  int j;
  int n = 1000;
  int iter = max_iter;
  float thresh = threshold;
  int divisor = 2;

  int opt, opt_index = 0;
  while((opt = getopt_long(argc, argv, "::n:i:t:q:", size_opts, &opt_index)) != -1){
    switch(opt){
      case 'n':
        n = atoi(optarg);
        break;
      case 'i':
        iter = atoi(optarg);
        break;
      case 't':
        thresh = atof(optarg);
        break;
      case 'q':
        divisor = atoi(optarg);
        break;
      default:
        usage(argv);
        exit(EXIT_FAILURE);
    }
  }
  
  size_t rank_size = (size_t)n * sizeof(float);
  size_t map_size = (size_t)n * n * sizeof(float);
  size_t page_size = (size_t)n * n * sizeof(int);
  size_t link_size = (size_t)n * sizeof(unsigned int);

  page_ranks = (float*)malloc(rank_size);
  maps = (float*)malloc(map_size);
  noutlinks = (unsigned int*)malloc(link_size);

  max_diff=max_diff_ref=99.0f;

  for (i=0; i<n; ++i) {
    noutlinks[i] = 0;
  }
  pages = random_pages(n,noutlinks,divisor);
  init_array(page_ranks, n, 1.0f / (float) n);

  float *diffs;
  diffs = (float*) malloc(rank_size);

  int *d_pages;
  float *d_maps;
  float *d_page_ranks;
  float *d_diffs;
  unsigned int *d_noutlinks;

  hipMalloc((void**)&d_pages, page_size);
  hipMalloc((void**)&d_page_ranks, rank_size);
  hipMalloc((void**)&d_maps, map_size);
  hipMalloc((void**)&d_noutlinks, link_size);
  hipMalloc((void**)&d_diffs, rank_size);
  hipMemset(d_diffs, 0, rank_size);

  hipMemcpy(d_pages, pages, page_size, hipMemcpyHostToDevice);
  hipMemcpy(d_page_ranks, page_ranks, rank_size, hipMemcpyHostToDevice);
  hipMemcpy(d_noutlinks, noutlinks, link_size, hipMemcpyHostToDevice);

  size_t block_size  = n < BLOCK_SIZE ? n : BLOCK_SIZE;
  size_t num_blocks = (n+block_size-1) / block_size;

  double ktime = 0.0;
  printf("Threshold: %f\n", thresh);
 
  for (t=1; t<=iter && max_diff>=thresh; ++t) {
    hipDeviceSynchronize();
    auto start = std::chrono::high_resolution_clock::now();

    map <<< dim3(num_blocks), dim3(block_size) >>> (
      d_pages, d_page_ranks, d_maps, d_noutlinks, n);

    reduce <<< dim3(num_blocks), dim3(block_size) >>> (
      d_page_ranks, d_maps, n, d_diffs);
    
    hipDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    ktime += std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();

    hipMemcpy(diffs, d_diffs, rank_size, hipMemcpyDeviceToHost);
    max_diff = maximum_dif(diffs, n);
  }
  //hipMemcpy(maps, d_maps, page_size, hipMemcpyDeviceToHost);
  //hipMemcpy(page_ranks, d_page_ranks, rank_size, hipMemcpyDeviceToHost);

  hipFree(d_pages);
  hipFree(d_maps);
  hipFree(d_page_ranks);
  hipFree(d_noutlinks);
  hipFree(d_diffs);

  fprintf(stderr, "Max difference %f is reached at iteration %d\n", max_diff, t);
  printf("\"Options\": \"-n %d -i %d -t %f\". Total kernel execution time: %lf (s)\n",
         n, iter, thresh, ktime);

  memset(diffs, 0, rank_size);
  for (t=1; t<=iter && max_diff_ref>=thresh; ++t) {
    map_ref(pages, page_ranks, maps, noutlinks, n);
    reduce_ref(page_ranks, maps, n, diffs);
    max_diff_ref = maximum_dif(diffs, n);
  }
  bool ok = fabsf(max_diff - max_diff_ref) < 1e-3f;
  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  free(pages);
  free(maps);
  free(page_ranks);
  free(noutlinks);
  free(diffs);

  return 0;
}
