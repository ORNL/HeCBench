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
#include <chrono>
#include <string.h>
#include <omp.h>
#include "../pathfinder-cuda/reference.h"

// halo width along one direction when advancing to the next iteration
#define HALO     1
#define NUMBER_THREADS 250
#define M_SEED   9
#define IN_RANGE(x, min, max)  ((x)>=(min) && (x)<=(max))
#define MIN(a, b) ((a)<=(b) ? (a) : (b))

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
  const int gws = (cols + smallBlockCol - 1) / smallBlockCol;

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

  const int size = rows * cols;
  // Teams cover the current row: overlapping blocks of width smallBlockCol,
  // not the full 2-D wall.
  int theHalo = HALO;

  auto start = std::chrono::steady_clock::now();

  // gpuWall is read-only in the kernel
  int* gpuWall = data+cols;
  // The "data" array should not be polluted, so allocate gpuSrc
  // and then copy part of the data array to gpuSrc
  int* gpuSrc = (int*) malloc (sizeof(int)*cols);
  int* gpuResult = (int*) malloc (sizeof(int)*cols);
  if (gpuSrc == NULL || gpuResult == NULL) {
    fprintf(stderr, "error: failed to allocate ping-pong buffers\n");
    exit(1);
  }
  memcpy(gpuSrc, data, cols*sizeof(int));

#pragma omp target data map(to: gpuSrc[0:cols]) \
                        map(alloc: gpuResult[0:cols]) \
                        map(to: gpuWall[0:size-cols])
  {
    auto kstart = std::chrono::steady_clock::now();

    for (int t = 0; t < rows - 1; t += pyramid_height)
    {
      // Calculate this for the kernel argument...
      int iteration = MIN(pyramid_height, rows-t-1);

      #pragma omp target teams num_teams(gws) thread_limit(NUMBER_THREADS)
      {
        // Ping-pong buffers for the pyramid: a step reads sm[cur] and writes
        // sm[1-cur], so no shared-memory copy-back is needed between steps.
        int sm[2][NUMBER_THREADS];
        #pragma omp parallel num_threads(NUMBER_THREADS)
        {
          // The ghost-zone decomposition uses a fixed logical block width, which
          // the host relies on to size the team count. The number of threads the
          // device actually provides may be smaller, so each thread strides over
          // the columns of its block.
          const int BLOCK_SIZE = NUMBER_THREADS;
          const int nthreads = omp_get_num_threads();
          const int bx = omp_get_team_num();
          const int tid = omp_get_thread_num();

          // Each block finally computes result for a small block
          // after N iterations.
          // it is the non-overlapping small blocks that cover
          // all the input data

          // calculate the small block size.
          const int small_block_cols = BLOCK_SIZE - (iteration*theHalo*2);

          // calculate the boundary for the block according to
          // the boundary of its small block
          const int blkX = (small_block_cols*bx) - borderCols;
          const int blkXmax = blkX+BLOCK_SIZE-1;

          // effective range within this block that falls within
          // the valid range of the input data
          // used to rule out computation outside the boundary.
          const int validXmin = (blkX < 0) ? -blkX : 0;
          const int validXmax = (blkXmax > cols-1) ? BLOCK_SIZE-1-(blkXmax-cols+1) : BLOCK_SIZE-1;

          for (int tx = tid; tx < BLOCK_SIZE; tx += nthreads)
          {
            if(IN_RANGE(tx, validXmin, validXmax))
            {
              sm[0][tx] = gpuSrc[blkX+tx];
            }
          }

          #pragma omp barrier

          int cur = 0;

          for (int i = 0; i < iteration; i++)
          {
            // The ghost zone loses one column on each side per pyramid step.
            const int lo = (i+1 > validXmin) ? i+1 : validXmin;
            const int hi = (BLOCK_SIZE-i-2 < validXmax) ? BLOCK_SIZE-i-2 : validXmax;
            // rows*cols is checked to fit in an int on the host.
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

            #pragma omp barrier
          }

          // update the global memory
          // after the last iteration, only the columns coordinated within the
          // small block were computed
          const int last = iteration-1;
          const int lo = (last+1 > validXmin) ? last+1 : validXmin;
          const int hi = (BLOCK_SIZE-last-2 < validXmax) ? BLOCK_SIZE-last-2 : validXmax;
          const int* __restrict__ result = sm[cur];

          for (int tx = lo + tid; tx <= hi; tx += nthreads)
          {
            gpuResult[blkX+tx] = result[tx];
          }
        }
      } 
      int *temp = gpuResult;
      gpuResult = gpuSrc;
      gpuSrc = temp;
    }

    // Final row lives in whichever host pointer gpuSrc currently names.
    #pragma omp target update from(gpuSrc[0:cols])

    auto kend = std::chrono::steady_clock::now();
    auto ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();
    printf("Total kernel execution time: %lf (s)\n", ktime * 1e-9);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Device offloading time = %lf (s)\n", time * 1e-9);

  memcpy(result, gpuSrc, sizeof(int)*cols);

  int* ref = new int[cols];
  PathfinderReference(data, rows, cols, ref);
  int unequal = memcmp(result, ref, sizeof(int) * cols);
  printf("%s\n", unequal ? "FAIL" : "PASS");

  // Memory cleanup here.
  delete[] data;
  delete[] wall;
  delete[] result;
  delete[] ref;
  free(gpuSrc);
  free(gpuResult);

  return EXIT_SUCCESS;
}
