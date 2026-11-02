/***********************************************************************
 * PathFinder uses dynamic programming to find a path on a 2-D grid from
 * the bottom row to the top row with the smallest accumulated weights,
 * where each step of the path moves straight ahead or diagonally ahead.
 * It iterates row by row, each node picks a neighboring node in the
 * previous row that has the smallest accumulated weight, and adds its
 * own weight to the sum.
 *
 * This kernel uses the technique of ghost zone optimization
 ***********************************************************************/

// Other header files.
#include <stdio.h>
#include <stdlib.h>
#include <climits>
#include <cstring>
#include <chrono>
#include <hip/hip_runtime.h>
#include "../pathfinder-cuda/reference.h"


// halo width along one direction when advancing to the next iteration
#define HALO     1
#define NUMBER_THREADS 250
#define M_SEED   9
#define IN_RANGE(x, min, max)  ((x)>=(min) && (x)<=(max))
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

__global__ void pathfinder (
    const int*__restrict__ gpuWall,
    const int*__restrict__ gpuSrc,
          int*__restrict__ gpuResult,
    const int iteration,
    const int theHalo,
    const int borderCols,
    const int cols,
    const int t)
{
  // Logical block width is fixed (host sizes the grid from NUMBER_THREADS).
  // Threads stride if the launch provides fewer than NUMBER_THREADS.
  const int BLOCK_SIZE = NUMBER_THREADS;
  const int nthreads = blockDim.x;
  const int bx = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ int sm[2][NUMBER_THREADS];

  // Each block finally computes result for a small block
  // after N iterations.
  // it is the non-overlapping small blocks that cover
  // all the input data

  const int small_block_cols = BLOCK_SIZE - (iteration*theHalo*2);
  const int blkX = (small_block_cols*bx) - borderCols;
  const int blkXmax = blkX+BLOCK_SIZE-1;

  const int validXmin = (blkX < 0) ? -blkX : 0;
  const int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;

  for (int tx = tid; tx < BLOCK_SIZE; tx += nthreads)
  {
    if (IN_RANGE(tx, validXmin, validXmax))
      sm[0][tx] = gpuSrc[blkX+tx];
  }

  __syncthreads();

  int cur = 0;
  for (int i = 0; i < iteration; i++)
  {
    const int lo = (i+1 > validXmin) ? i+1 : validXmin;
    const int hi = (BLOCK_SIZE-i-2 < validXmax) ? BLOCK_SIZE-i-2 : validXmax;
    const int* __restrict__ wallRow = gpuWall + (cols*(t+i) + blkX);
    const int* __restrict__ prev = sm[cur];
    int* __restrict__ result = sm[cur^1];

    for (int tx = lo + tid; tx <= hi; tx += nthreads)
    {
      const int W = (tx-1 < validXmin) ? validXmin : tx-1;
      const int E = (tx+1 > validXmax) ? validXmax : tx+1;
      int left = prev[W];
      int up = prev[tx];
      int right = prev[E];
      int shortest = MIN(left, up);
      shortest = MIN(shortest, right);
      result[tx] = shortest + wallRow[tx];
    }

    cur ^= 1;
    __syncthreads();
  }

  const int last = iteration-1;
  const int lo = (last+1 > validXmin) ? last+1 : validXmin;
  const int hi = (BLOCK_SIZE-last-2 < validXmax) ? BLOCK_SIZE-last-2 : validXmax;
  const int* __restrict__ result = sm[cur];
  for (int tx = lo + tid; tx <= hi; tx += nthreads)
    gpuResult[blkX+tx] = result[tx];
}

int main(int argc, char** argv)
{
  // Program variables.
  int   rows, cols;
  int*  data;
  int** wall;
  int*  result;
  int   pyramid_height;

  if (argc == 4)
  {
    cols = atoi(argv[1]);
    rows = atoi(argv[2]);
    pyramid_height = atoi(argv[3]);
  }
  else
  {
    printf("Usage: %s <column length> <row length> <pyramid_height>\n", argv[0]);
    exit(0);
  }

  if (rows < 1 || cols < 1 || pyramid_height < 1) {
    fprintf(stderr, "error: rows, cols, and pyramid_height must be positive\n");
    exit(1);
  }
  if (rows > INT_MAX / cols) {
    fprintf(stderr, "error: rows * cols exceeds INT_MAX\n");
    exit(1);
  }

  const int lws = NUMBER_THREADS;
  int smallBlockCol = lws - pyramid_height * HALO * 2;
  if (smallBlockCol < 1) {
    fprintf(stderr,
            "error: pyramid_height (%d) too large for work-group size %d\n",
            pyramid_height, lws);
    exit(1);
  }
  int blockCols = (cols + smallBlockCol - 1) / smallBlockCol;

  data = new int[rows * cols];
  wall = new int*[rows];
  for (int n = 0; n < rows; n++)
  {
    // wall[n] is set to be the nth row of the data array.
    wall[n] = data + cols * n;
  }
  result = new int[cols];

  int seed = M_SEED;
  srand(seed);

  for (int i = 0; i < rows; i++)
  {
    for (int j = 0; j < cols; j++)
    {
      wall[i][j] = rand() % 10;
    }
  }

  // Pyramid parameters.
  const int borderCols = (pyramid_height) * HALO;

  /* printf("pyramidHeight: %d\ngridSize: [%d]\nborder:[%d]\nblockSize: %d\nblockGrid:[%d]\ntargetBlock:[%d]\n",
     pyramid_height, cols, borderCols, NUMBER_THREADS, blockCols, smallBlockCol); */

  int size = rows * cols;

  // Grid covers the current row: overlapping blocks of width smallBlockCol,
  // not the full 2-D wall.
  int theHalo = HALO;

  auto start = std::chrono::steady_clock::now();

  int* d_gpuWall;
  hipMalloc((void**)&d_gpuWall, sizeof(int)*(size-cols));
  hipMemcpy(d_gpuWall, data+cols, sizeof(int)*(size-cols), hipMemcpyHostToDevice);

  int* d_gpuSrc;
  hipMalloc((void**)&d_gpuSrc, sizeof(int)*cols);
  hipMemcpy(d_gpuSrc, data, sizeof(int)*cols, hipMemcpyHostToDevice);

  int* d_gpuResult;
  hipMalloc((void**)&d_gpuResult, sizeof(int)*cols);

  dim3 gridDim (blockCols);
  dim3 blockDim (lws);

  hipDeviceSynchronize();
  auto kstart = std::chrono::steady_clock::now();

  for (int t = 0; t < rows - 1; t += pyramid_height)
  {
    // Calculate this for the kernel argument...
    int iteration = MIN(pyramid_height, rows-t-1);

    pathfinder<<<gridDim, blockDim>>>(
        d_gpuWall, d_gpuSrc, d_gpuResult,
        iteration, theHalo, borderCols, cols, t);

    int* temp = d_gpuResult;
    d_gpuResult = d_gpuSrc;
    d_gpuSrc = temp;
  }

  hipDeviceSynchronize();
  auto kend = std::chrono::steady_clock::now();
  auto ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();
  printf("Total kernel execution time: %lf (s)\n", ktime * 1e-9);

  hipMemcpy(result, d_gpuSrc, sizeof(int)*cols, hipMemcpyDeviceToHost);

  hipFree(d_gpuResult);
  hipFree(d_gpuSrc);
  hipFree(d_gpuWall);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Device offloading time = %lf (s)\n", time * 1e-9);

  int* ref = new int[cols];
  PathfinderReference(data, rows, cols, ref);
  int unequal = memcmp(result, ref, sizeof(int) * cols);
  printf("%s\n", unequal ? "FAIL" : "PASS");

  // Memory cleanup here.
  delete[] data;
  delete[] wall;
  delete[] result;
  delete[] ref;

  return EXIT_SUCCESS;
}
