/*
TSP_GPU: This code is a GPU-accelerated heuristic solver for the
symmetric Traveling Salesman Problem that is based on iterative hill
climbing with 2-opt local search.

Copyright (c) 2014-2020, Texas State University. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright
 notice, this list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright
 notice, this list of conditions and the following disclaimer in the
 documentation and/or other materials provided with the distribution.
 * Neither the name of Texas State University nor the names of its
 contributors may be used to endorse or promote products derived from
 this software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL TEXAS STATE UNIVERSITY BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Authors: Martin Burtscher

URL: The latest version of this code is available at
https://userweb.cs.txstate.edu/~burtscher/research/TSP_GPU/.

Publication: This work is described in detail in the following paper.
Molly A. O'Neil and Martin Burtscher. Rethinking the Parallelization of
Random-Restart Hill Climbing. Proceedings of the Eighth Workshop on General
Purpose Processing Using GPUs (10 pages). February 2015.
 */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <chrono>
#include <hip/hip_runtime.h>

// no point in using precise FP math or double precision as we are rounding
// the results to the nearest integer anyhow

/******************************************************************************/
/*** 2-opt with random restarts ***********************************************/
/******************************************************************************/

#define tilesize 128
#define dist(a, b) int(sqrtf((px[a] - px[b]) * (px[a] - px[b]) + (py[a] - py[b]) * (py[a] - py[b])))
#define swap(a, b) {float tmp = a;  a = b;  b = tmp;}

__device__
float LCG_random(unsigned int * seed) {
  const unsigned int m = 2147483648;
  const unsigned int a = 26757677;
  const unsigned int c = 1;
  *seed = (a * (*seed) + c) % m;
  return (float) (*seed) / (float) m;
}

__global__
void TwoOpt(int cities, 
    const float *__restrict posx_d,
    const float *__restrict posy_d,
    int *__restrict glob_d,
    int *__restrict climbs_d,
    int *__restrict best_d)
{
  extern __shared__ int buf_s[];
  __shared__ float px_s[tilesize];
  __shared__ float py_s[tilesize];
  __shared__ int bf_s[tilesize];

  int *buf = &glob_d[blockIdx.x * ((3 * cities + 2 + 31) / 32 * 32)];
  float *px = (float *)(&buf[cities]);
  float *py = &px[cities + 1];

  for (int i = threadIdx.x; i < cities; i += blockDim.x) px[i] = posx_d[i];
  for (int i = threadIdx.x; i < cities; i += blockDim.x) py[i] = posy_d[i];
  __syncthreads();

  if (threadIdx.x == 0) {  // serial permutation
    unsigned int seed = blockIdx.x;
    for (unsigned int i = 1; i < cities; i++) {
      int j = (int)(LCG_random(&seed) * (cities - 1)) + 1;
      swap(px[i], px[j]);
      swap(py[i], py[j]);
    }
    px[cities] = px[0];
    py[cities] = py[0];
  }
  __syncthreads();

  int minchange;
  do {
    for (int i = threadIdx.x; i < cities; i += blockDim.x) buf[i] = -dist(i, i + 1);
    __syncthreads();

    minchange = 0;
    int mini = 1;
    int minj = 0;
    for (int ii = 0; ii < cities - 2; ii += blockDim.x) {
      int i = ii + threadIdx.x;
      float pxi0, pyi0, pxi1, pyi1, pxj1, pyj1;
      if (i < cities - 2) {
        minchange -= buf[i];
        pxi0 = px[i];
        pyi0 = py[i];
        pxi1 = px[i + 1];
        pyi1 = py[i + 1];
        pxj1 = px[cities];
        pyj1 = py[cities];
      }
      for (int jj = cities - 1; jj >= ii + 2; jj -= tilesize) {
        int bound = jj - tilesize + 1;
        for (int k = threadIdx.x; k < tilesize; k += blockDim.x) {
          if (k + bound >= ii + 2) {
            px_s[k] = px[k + bound];
            py_s[k] = py[k + bound];
            bf_s[k] = buf[k + bound];
          }
        }
        __syncthreads();

        int lower = bound;
        if (lower < i + 2) lower = i + 2;
        for (int j = jj; j >= lower; j--) {
          int jm = j - bound;
          float pxj0 = px_s[jm];
          float pyj0 = py_s[jm];
          int change = bf_s[jm]
            + int(sqrtf((pxi0 - pxj0) * (pxi0 - pxj0) + (pyi0 - pyj0) * (pyi0 - pyj0)))
            + int(sqrtf((pxi1 - pxj1) * (pxi1 - pxj1) + (pyi1 - pyj1) * (pyi1 - pyj1)));
          pxj1 = pxj0;
          pyj1 = pyj0;
          if (minchange > change) {
            minchange = change;
            mini = i;
            minj = j;
          }
        }
        __syncthreads();
      }

      if (i < cities - 2) {
        minchange += buf[i];
      }
    }
    __syncthreads();

    int change = buf_s[threadIdx.x] = minchange;
    if (threadIdx.x == 0) atomicAdd(climbs_d, 1);  // stats only
    __syncthreads();

    int j = blockDim.x;
    do {
      int k = (j + 1) / 2;
      if ((threadIdx.x + k) < j) {
        int tmp = buf_s[threadIdx.x + k];
        if (change > tmp) change = tmp;
        buf_s[threadIdx.x] = change;
      }
      j = k;
      __syncthreads();
    } while (j > 1);

    if (minchange == buf_s[0]) {
      buf_s[1] = threadIdx.x;  // non-deterministic winner
    }
    __syncthreads();

    if (threadIdx.x == buf_s[1]) {
      buf_s[2] = mini + 1;
      buf_s[3] = minj;
    }
    __syncthreads();

    minchange = buf_s[0];
    mini = buf_s[2];
    int sum = buf_s[3] + mini;
    for (int i = threadIdx.x; (i + i) < sum; i += blockDim.x) {
      if (mini <= i) {
        int j = sum - i;
        swap(px[i], px[j]);
        swap(py[i], py[j]);
      }
    }
    __syncthreads();
  } while (minchange < 0);

  int term = 0;
  for (int i = threadIdx.x; i < cities; i += blockDim.x) {
    term += dist(i, i + 1);
  }
  buf_s[threadIdx.x] = term;
  __syncthreads();

  int j = blockDim.x;
  do {
    int k = (j + 1) / 2;
    if ((threadIdx.x + k) < j) {
      term += buf_s[threadIdx.x + k];
    }
    __syncthreads();
    if ((threadIdx.x + k) < j) {
      buf_s[threadIdx.x] = term;
    }
    j = k;
    __syncthreads();
  } while (j > 1);

  if (threadIdx.x == 0) {
    atomicMin(best_d, term);
  }
}

/******************************************************************************/
/*** find best thread count ***************************************************/
/******************************************************************************/

static int best_thread_count(int cities)
{
  int max, best, threads, smem, blocks, thr, perf, bthr;

  max = cities - 2;
  if (max > 256) max = 256;
  best = 0;
  bthr = 4;
  for (threads = 1; threads <= max; threads++) {
    smem = sizeof(int) * threads + 2 * sizeof(float) * tilesize + sizeof(int) * tilesize;
    blocks = (16384 * 2) / smem;
    if (blocks > 16) blocks = 16;
    thr = (threads + 31) / 32 * 32;
    while (blocks * thr > 2048) blocks--;
    perf = threads * blocks;
    if (perf > best) {
      best = perf;
      bthr = threads;
    }
  }

  return bthr;
}

int main(int argc, char *argv[])
{
  printf("2-opt TSP HIP GPU code v2.3\n");
  printf("Copyright (c) 2014-2020, Texas State University. All rights reserved.\n");

  if (argc != 4) {
    fprintf(stderr, "\narguments: <input_file> <restart_count> <repeat>\n");
    exit(-1);
  }

  FILE *f = fopen(argv[1], "rt");
  if (f == NULL) {fprintf(stderr, "could not open file %s\n", argv[1]);  exit(-1);}

  int restarts = atoi(argv[2]);
  if (restarts < 1) {fprintf(stderr, "restart_count is too small: %d\n", restarts); exit(-1);}

  int repeat = atoi(argv[3]);

  //======================================================================
  // read data from input file
  //======================================================================
  int ch, in1;
  float in2, in3;
  char str[256];

  ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
  ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
  ch = getc(f);  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
  ch = getc(f);  while ((ch != EOF) && (ch != ':')) ch = getc(f);
  fscanf(f, "%s\n", str);

  int cities = atoi(str);
  if (cities < 100) {
    fprintf(stderr, "the problem size must be at least 100 for this version of the code\n");
    fclose(f);
    exit(-1);
  } 

  ch = getc(f); 
  while ((ch != EOF) && (ch != '\n')) ch = getc(f);
  fscanf(f, "%s\n", str);
  if (strcmp(str, "NODE_COORD_SECTION") != 0) {
    fprintf(stderr, "wrong file format\n");
    fclose(f);
    exit(-1);
  }

  float *posx = (float *)malloc(sizeof(float) * cities);
  if (posx == NULL) fprintf(stderr, "cannot allocate posx\n");
  float *posy = (float *)malloc(sizeof(float) * cities);
  if (posy == NULL) fprintf(stderr, "cannot allocate posy\n");

  int cnt = 0;
  while (fscanf(f, "%d %f %f\n", &in1, &in2, &in3)) {
    posx[cnt] = in2;
    posy[cnt] = in3;
    cnt++;
    if (cnt > cities) fprintf(stderr, "input too long\n");
    if (cnt != in1) fprintf(stderr, "input line mismatch: expected %d instead of %d\n", cnt, in1);
  }
  if (cnt != cities) fprintf(stderr, "read %d instead of %d cities\n", cnt, cities);

  fscanf(f, "%s", str);
  if (strcmp(str, "EOF") != 0) fprintf(stderr, "didn't see 'EOF' at end of file\n");

  fclose(f);

  printf("configuration: %d cities, %d restarts, %s input\n", cities, restarts, argv[1]);

  //======================================================================
  // device region
  //======================================================================
  float *posx_d, *posy_d;
  int *glob_d;
  int *climbs_d;
  int *best_d;
  int climbs = 0;
  int best = INT_MAX;

  hipMalloc((void**)&climbs_d, sizeof(int));
  hipMalloc((void**)&best_d, sizeof(int));
  hipMalloc((void**)&posx_d, sizeof(float) * cities);
  hipMalloc((void**)&posy_d, sizeof(float) * cities);
  hipMalloc((void**)&glob_d, sizeof(int) * restarts * ((3 * cities + 2 + 31) / 32 * 32));

  hipMemcpy(posx_d, posx, sizeof(float) * cities, hipMemcpyHostToDevice);
  hipMemcpy(posy_d, posy, sizeof(float) * cities, hipMemcpyHostToDevice);

  int threads = best_thread_count(cities);
  printf("thread block size: %d\n", threads);

  double ktime = 0.0;

  for (int i = 0; i <= repeat; i++) {
    hipMemcpy(climbs_d, &climbs, sizeof(int), hipMemcpyHostToDevice);
    hipMemcpy(best_d, &best, sizeof(int), hipMemcpyHostToDevice);

    auto kstart = std::chrono::steady_clock::now();

    TwoOpt<<<restarts, threads, sizeof(int) * threads>>>(cities, posx_d, posy_d, glob_d, climbs_d, best_d);

    hipDeviceSynchronize();
    auto kend = std::chrono::steady_clock::now();
    if (i > 0)
      ktime += std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();
  }

  hipMemcpy(&best, best_d, sizeof(int), hipMemcpyDeviceToHost);
  hipMemcpy(&climbs, climbs_d, sizeof(int), hipMemcpyDeviceToHost);

  long long moves = 1LL * climbs * (cities - 2) * (cities - 1) / 2;

  printf("Average kernel time: %.4f s\n", ktime * 1e-9f / repeat);
  printf("%.3f Gmoves/s\n", moves * repeat / ktime);
  printf("Best found tour length is %d with %d climbers\n", best, climbs);

  // for the specific dataset d493.tsp
  if (best < 38000 && best >= 35002)
    printf("PASS\n");
  else
    printf("FAIL\n");

  hipFree(glob_d);
  hipFree(best_d);
  hipFree(climbs_d);
  hipFree(posx_d);
  hipFree(posy_d);
  free(posx);
  free(posy);
  if (best < 38000 && best < 35002) return 1;
  return 0;
}
