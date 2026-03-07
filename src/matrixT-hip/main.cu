// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.

// ----------------------------------------------------------------------------------------
// Transpose
//
// This file contains both device and host code for transposing a floating-point
// matrix.  It performs several transpose kernels, which incrementally improve performance
// through coalescing, removing shared memory bank conflicts, and eliminating partition
// camping.  Several of the kernels perform a copy, used to represent the best case
// performance that a transpose can achieve.
// ----------------------------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <chrono>
#include <hip/hip_cooperative_groups.h>

#define checkHipErrors(call)                                                          \
  do {                                                                                \
    hipError_t err = call;                                                            \
    if (err != hipSuccess) {                                                          \
      printf("HIP error at %s %d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
    }                                                                                 \
  } while (0)

namespace cg = cooperative_groups;

// Utilities and system includes

// Each block transposes/copies a tile of TILE_DIM x TILE_DIM elements
// using TILE_DIM x BLOCK_ROWS threads, so that each thread transposes
// TILE_DIM/BLOCK_ROWS elements.  TILE_DIM must be an integral multiple of BLOCK_ROWS

#define TILE_DIM    16
#define BLOCK_ROWS  16

// This sample assumes that MATRIX_SIZE_X = MATRIX_SIZE_Y
int MATRIX_SIZE_X = 1024;
int MATRIX_SIZE_Y = 1024;
int MUL_FACTOR    = TILE_DIM;

#define FLOOR(a,b) (a-(a%b))

// Compute the tile size necessary to illustrate performance cases for SM20+ hardware
int MAX_TILES = (FLOOR(MATRIX_SIZE_X,512) * FLOOR(MATRIX_SIZE_Y,512)) / (TILE_DIM *TILE_DIM);

// -------------------------------------------------------
// Copies
// width and height must be integral multiples of TILE_DIM
// -------------------------------------------------------

__global__ void copy(
        float *__restrict__ odata,
  const float *__restrict__ idata,
  int width, int height)
{
  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

  int index  = xIndex + width*yIndex;

  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
  {
    odata[index+i*width] = idata[index+i*width];
  }
}

__global__ void copySharedMem(
        float *__restrict__ odata,
  const float *__restrict__ idata,
  int width, int height)
{
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ float tile[TILE_DIM][TILE_DIM];

  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

  int index  = xIndex + width*yIndex;

  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
  {
    if (xIndex < width && yIndex < height)
    {
      tile[threadIdx.y][threadIdx.x] = idata[index];
    }
  }

  cg::sync(cta);

  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
  {
    if (xIndex < height && yIndex < width)
    {
      odata[index] = tile[threadIdx.y][threadIdx.x];
    }
  }
}

// -------------------------------------------------------
// Transposes
// width and height must be integral multiples of TILE_DIM
// -------------------------------------------------------

__global__ void transposeNaive(
        float *__restrict__ odata,
  const float *__restrict__ idata,
  int width, int height)
{
  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

  int index_in  = xIndex + width * yIndex;
  int index_out = yIndex + height * xIndex;

  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
  {
    odata[index_out+i] = idata[index_in+i*width];
  }
}

// coalesced transpose (with bank conflicts)

__global__ void transposeCoalesced(
        float *__restrict__ odata,
  const float *__restrict__ idata,
  int width, int height)
{
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ float tile[TILE_DIM][TILE_DIM];

  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex)*height;

  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
  {
    tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
  }

  cg::sync(cta);

  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
  {
    odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
  }
}

// Coalesced transpose with no bank conflicts

__global__ void transposeNoBankConflicts(
        float *__restrict__ odata,
  const float *__restrict__ idata,
  int width, int height)
{
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ float tile[TILE_DIM][TILE_DIM+1];

  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex)*height;

  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
  {
    tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
  }

  cg::sync(cta);

  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
  {
    odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
  }
}

// Transpose that effectively reorders execution of thread blocks along diagonals of the
// matrix (also coalesced and has no bank conflicts)
//
// Here blockIdx.x is interpreted as the distance along a diagonal and blockIdx.y as
// corresponding to different diagonals
//
// blockIdx_x and blockIdx_y expressions map the diagonal coordinates to the more commonly
// used cartesian coordinates so that the only changes to the code from the coalesced version
// are the calculation of the blockIdx_x and blockIdx_y and replacement of blockIdx.x and
// bloclIdx.y with the subscripted versions in the remaining code

__global__ void transposeDiagonal(
        float *__restrict__ odata,
  const float *__restrict__ idata,
  int width, int height)
{
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ float tile[TILE_DIM][TILE_DIM+1];

  int blockIdx_x, blockIdx_y;

  // do diagonal reordering
  if (width == height)
  {
    blockIdx_y = blockIdx.x;
    blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
  }
  else
  {
    int bid = blockIdx.x + gridDim.x*blockIdx.y;
    blockIdx_y = bid%gridDim.y;
    blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
  }

  // from here on the code is same as previous kernel except blockIdx_x replaces blockIdx.x
  // and similarly for y

  int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx_y * TILE_DIM + threadIdx.y;
  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx_y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx_x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex)*height;

  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
  {
    tile[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
  }

  cg::sync(cta);

  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
  {
    odata[index_out+i*height] = tile[threadIdx.x][threadIdx.y+i];
  }
}

// --------------------------------------------------------------------
// Partial transposes
// NB: the coarse- and fine-grained routines only perform part of a
//     transpose and will fail the test against the reference solution
//
//     They are used to assess performance characteristics of different
//     components of a full transpose
// --------------------------------------------------------------------

__global__ void transposeFineGrained(
        float *__restrict__ odata,
  const float *__restrict__ idata,
  int width, int height)
{
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ float block[TILE_DIM][TILE_DIM+1];

  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
  int index = xIndex + (yIndex)*width;

  for (int i=0; i < TILE_DIM; i += BLOCK_ROWS)
  {
    block[threadIdx.y+i][threadIdx.x] = idata[index+i*width];
  }

  cg::sync(cta);

  for (int i=0; i < TILE_DIM; i += BLOCK_ROWS)
  {
    odata[index+i*height] = block[threadIdx.x][threadIdx.y+i];
  }
}

__global__ void transposeCoarseGrained(
        float *__restrict__ odata,
  const float *__restrict__ idata,
  int width, int height)
{
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  __shared__ float block[TILE_DIM][TILE_DIM+1];

  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex)*height;

  for (int i=0; i<TILE_DIM; i += BLOCK_ROWS)
  {
    block[threadIdx.y+i][threadIdx.x] = idata[index_in+i*width];
  }

  cg::sync(cta);

  for (int i=0; i<TILE_DIM; i += BLOCK_ROWS)
  {
    odata[index_out+i*height] = block[threadIdx.y+i][threadIdx.x];
  }
}

// ---------------------
// host utility routines
// ---------------------

void computeTransposeGold(float *gold, float *idata,
    const  int size_x, const  int size_y)
{
  for (int y = 0; y < size_y; ++y)
  {
    for (int x = 0; x < size_x; ++x)
    {
      gold[(x * size_y) + y] = idata[(y * size_x) + x];
    }
  }
}

void showHelp()
{
  printf("\nCommand line options\n");
  printf("\t<row_dim_size> (matrix row    dimensions)\n");
  printf("\t<col_dim_size> (matrix column dimensions)\n");
  printf("\t<repeat> (kernel execution count)\n");
}

int main(int argc, char **argv)
{
  if (argc != 4)
  {
    showHelp();
    return 0;
  }

  // Calculate number of tiles we will run for the Matrix Transpose performance tests
  int size_x = atoi(argv[1]);
  int size_y = atoi(argv[2]);
  int repeat = atoi(argv[3]);

  if (size_x != size_y)
  {
    printf("Error: non-square matrices (row_dim_size(%d) != col_dim_size(%d))\nExiting...\n\n", size_x, size_y);
    exit(EXIT_FAILURE);
  }

  if (size_x%TILE_DIM != 0 || size_y%TILE_DIM != 0)
  {
    printf("Matrix size must be integral multiple of tile size\nExiting...\n\n");
    exit(EXIT_FAILURE);
  }

  // kernel pointer and descriptor
  void (*kernel)(float *__restrict__, const float *__restrict__, int, int);
  const char *kernelName;

  // execution configuration parameters
  dim3 grid(size_x/TILE_DIM, size_y/TILE_DIM);
  dim3 threads(TILE_DIM,BLOCK_ROWS);

  if (grid.x < 1 || grid.y < 1)
  {
    printf("grid size computation incorrect in test \nExiting...\n\n");
    exit(EXIT_FAILURE);
  }

  // size of memory required to store the matrix
  size_t mem_size = static_cast<size_t>(sizeof(float) * size_x*size_y);

  // allocate host memory
  float *h_idata = (float *) malloc(mem_size);
  float *h_odata = (float *) malloc(mem_size);
  float *transposeGold = (float *) malloc(mem_size);
  float *gold;

  // allocate device memory
  float *d_idata, *d_odata;
  checkHipErrors(hipMalloc((void **) &d_idata, mem_size));
  checkHipErrors(hipMalloc((void **) &d_odata, mem_size));

  // initialize host data
  for (int i = 0; i < (size_x*size_y); ++i)
  {
    h_idata[i] = (float) i;
  }

  // copy host data to device
  checkHipErrors(hipMemcpy(d_idata, h_idata, mem_size, hipMemcpyHostToDevice));

  // Compute reference transpose solution
  computeTransposeGold(transposeGold, h_idata, size_x, size_y);

  // print out common data for all kernels
  printf("\nMatrix size: %dx%d (%dx%d tiles), tile size: %dx%d, block size: %dx%d\n\n",
      size_x, size_y, size_x/TILE_DIM, size_y/TILE_DIM, TILE_DIM, TILE_DIM, TILE_DIM, BLOCK_ROWS);

  //
  // loop over different kernels
  //
  bool success = true;

  for (int k = 0; k<8; k++)
  {
    // set kernel pointer
    switch (k)
    {
      case 0:
        kernel = &copy;
        kernelName = "simple copy       ";
        break;

      case 1:
        kernel = &copySharedMem;
        kernelName = "shared memory copy";
        break;

      case 2:
        kernel = &transposeNaive;
        kernelName = "naive             ";
        break;

      case 3:
        kernel = &transposeCoalesced;
        kernelName = "coalesced         ";
        break;

      case 4:
        kernel = &transposeNoBankConflicts;
        kernelName = "optimized         ";
        break;

      case 5:
        kernel = &transposeCoarseGrained;
        kernelName = "coarse-grained    ";
        break;

      case 6:
        kernel = &transposeFineGrained;
        kernelName = "fine-grained      ";
        break;

      case 7:
        kernel = &transposeDiagonal;
        kernelName = "diagonal          ";
        break;
    }

    // set reference solution
    if (kernel == &copy || kernel == &copySharedMem)
    {
      gold = h_idata;
    }
    else if (kernel == &transposeCoarseGrained || kernel == &transposeFineGrained)
    {
      gold = h_odata;   // fine- and coarse-grained kernels are not full transposes, so bypass check
    }
    else
    {
      gold = transposeGold;
    }

    hipDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    for (int i=0; i < repeat; i++)
    {
      hipLaunchKernelGGL(kernel, grid, threads, 0, 0, d_odata, d_idata, size_x, size_y);
    }

    hipDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel (%s) execution time: %f (us)\n", kernelName, (time * 1e-3f) / repeat);

    checkHipErrors(hipMemcpy(h_odata, d_odata, mem_size, hipMemcpyDeviceToHost));
 
    bool ok = true;
    for (int i = 0; i < size_x*size_y; i++)
      if (fabsf(gold[i] - h_odata[i]) > 0.01f) {
        ok = false;
        break;
      }

    if (!ok)
    {
      printf("*** %s kernel FAILED ***\n", kernelName);
      success = false;
    }
  }

  printf("%s\n", success ? "PASS" : "FAIL");
  if (!success) exit(1);

  // cleanup
  free(h_idata);
  free(h_odata);
  free(transposeGold);
  hipFree(d_idata);
  hipFree(d_odata);

  return 0;
}
