// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <sycl/sycl.hpp>

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

#define __syncthreads() \
  item.barrier(sycl::access::fence_space::local_space)

// -------------------------------------------------------
// Copies
// width and height must be integral multiples of TILE_DIM
// -------------------------------------------------------

void copy(
  sycl::nd_item<2> &item,
        float *__restrict odata,
  const float *__restrict idata,
  int width, int height)
{
  int xIndex = item.get_global_id(1);
  int yIndex = item.get_global_id(0);

  int index  = xIndex + width*yIndex;

  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
  {
    odata[index+i*width] = idata[index+i*width];
  }
}

void copySharedMem(
  sycl::nd_item<2> &item,
        float *__restrict tile,
        float *__restrict odata,
  const float *__restrict idata,
  int width, int height)
{
  int threadIdx_x = item.get_local_id(1);
  int threadIdx_y = item.get_local_id(0);

  int xIndex = item.get_group(1) * TILE_DIM + threadIdx_x;
  int yIndex = item.get_group(0) * TILE_DIM + threadIdx_y;

  int index  = xIndex + width*yIndex;

  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
  {
    if (xIndex < width && yIndex < height)
    {
      tile[threadIdx_y * TILE_DIM + threadIdx_x] = idata[index];
    }
  }

  __syncthreads();

  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
  {
    if (xIndex < height && yIndex < width)
    {
      odata[index] = tile[threadIdx_y * TILE_DIM + threadIdx_x];
    }
  }
}

// -------------------------------------------------------
// Transposes
// width and height must be integral multiples of TILE_DIM
// -------------------------------------------------------

void transposeNaive(
  sycl::nd_item<2> &item,
        float *__restrict odata,
  const float *__restrict idata,
  int width, int height)
{
  int threadIdx_x = item.get_local_id(1);
  int threadIdx_y = item.get_local_id(0);

  int xIndex = item.get_group(1) * TILE_DIM + threadIdx_x;
  int yIndex = item.get_group(0) * TILE_DIM + threadIdx_y;

  int index_in  = xIndex + width * yIndex;
  int index_out = yIndex + height * xIndex;

  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
  {
    odata[index_out+i] = idata[index_in+i*width];
  }
}

// coalesced transpose (with bank conflicts)

void transposeCoalesced(
  sycl::nd_item<2> &item,
        float *__restrict tile,
        float *__restrict odata,
  const float *__restrict idata,
  int width, int height)
{
  int threadIdx_x = item.get_local_id(1);
  int threadIdx_y = item.get_local_id(0);
  int blockIdx_x = item.get_group(1);
  int blockIdx_y = item.get_group(0);

  int xIndex = blockIdx_x * TILE_DIM + threadIdx_x;
  int yIndex = blockIdx_y * TILE_DIM + threadIdx_y;
  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx_y * TILE_DIM + threadIdx_x;
  yIndex = blockIdx_x * TILE_DIM + threadIdx_y;
  int index_out = xIndex + (yIndex)*height;

  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
  {
    tile[(threadIdx_y+i)*TILE_DIM+threadIdx_x] = idata[index_in+i*width];
  }

  __syncthreads();

  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
  {
    odata[index_out+i*height] = tile[threadIdx_x * TILE_DIM + threadIdx_y+i];
  }
}

// Coalesced transpose with no bank conflicts

void transposeNoBankConflicts(
  sycl::nd_item<2> &item,
        float *__restrict tile,
        float *__restrict odata,
  const float *__restrict idata,
  int width, int height)
{
  int threadIdx_x = item.get_local_id(1);
  int threadIdx_y = item.get_local_id(0);
  int blockIdx_x = item.get_group(1);
  int blockIdx_y = item.get_group(0);

  int xIndex = blockIdx_x * TILE_DIM + threadIdx_x;
  int yIndex = blockIdx_y * TILE_DIM + threadIdx_y;
  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx_y * TILE_DIM + threadIdx_x;
  yIndex = blockIdx_x * TILE_DIM + threadIdx_y;
  int index_out = xIndex + (yIndex)*height;

  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
  {
    tile[(threadIdx_y+i)*(TILE_DIM+1)+threadIdx_x] = idata[index_in+i*width];
  }

  __syncthreads();

  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
  {
    odata[index_out+i*height] = tile[threadIdx_x*(TILE_DIM+1)+threadIdx_y+i];
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

void transposeDiagonal(
  sycl::nd_item<2> &item,
        float *__restrict tile,
        float *__restrict odata,
  const float *__restrict idata,
  int width, int height)
{
  int bidx = item.get_group(1);
  int bidy = item.get_group(0);
  int gridDim_x = item.get_group_range(1);
  int gridDim_y = item.get_group_range(0);
  int threadIdx_x = item.get_local_id(1);
  int threadIdx_y = item.get_local_id(0);

  // do diagonal reordering
  int blockIdx_x , blockIdx_y;
  if (width == height)
  {
    blockIdx_y = bidx;
    blockIdx_x = (bidx+bidy)%gridDim_x;
  }
  else
  {
    int bid = bidx + gridDim_x*bidy;
    blockIdx_y = bid%gridDim_y;
    blockIdx_x = ((bid/gridDim_y)+blockIdx_y)%gridDim_x;
  }

  // from here on the code is same as previous kernel except blockIdx_x replaces blockIdx.x
  // and similarly for y

  int xIndex = blockIdx_x * TILE_DIM + threadIdx_x;
  int yIndex = blockIdx_y * TILE_DIM + threadIdx_y;
  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx_y * TILE_DIM + threadIdx_x;
  yIndex = blockIdx_x * TILE_DIM + threadIdx_y;
  int index_out = xIndex + (yIndex)*height;

  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
  {
    tile[(threadIdx_y+i)*(TILE_DIM+1)+threadIdx_x] = idata[index_in+i*width];
  }

  __syncthreads();

  for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS)
  {
    odata[index_out+i*height] = tile[threadIdx_x*(TILE_DIM+1)+threadIdx_y+i];
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

void transposeFineGrained(
  sycl::nd_item<2> &item,
        float *__restrict tile,
        float *__restrict odata,
  const float *__restrict idata,
  int width, int height)
{
  int blockIdx_x = item.get_group(1);
  int blockIdx_y = item.get_group(0);
  int threadIdx_x = item.get_local_id(1);
  int threadIdx_y = item.get_local_id(0);

  int xIndex = blockIdx_x * TILE_DIM + threadIdx_x;
  int yIndex = blockIdx_y * TILE_DIM + threadIdx_y;
  int index = xIndex + (yIndex)*width;

  for (int i=0; i < TILE_DIM; i += BLOCK_ROWS)
  {
    tile[(threadIdx_y+i)*(TILE_DIM+1)+threadIdx_x] = idata[index+i*width];
  }

  __syncthreads();

  for (int i=0; i < TILE_DIM; i += BLOCK_ROWS)
  {
    odata[index+i*height] = tile[threadIdx_x*(TILE_DIM+1)+threadIdx_y+i];
  }
}


void transposeCoarseGrained(
  sycl::nd_item<2> &item,
        float *__restrict tile,
        float *__restrict odata,
  const float *__restrict idata,
  int width, int height)
{
  int threadIdx_x = item.get_local_id(1);
  int threadIdx_y = item.get_local_id(0);
  int blockIdx_x = item.get_group(1);
  int blockIdx_y = item.get_group(0);

  int xIndex = blockIdx_x * TILE_DIM + threadIdx_x;
  int yIndex = blockIdx_y * TILE_DIM + threadIdx_y;
  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx_y * TILE_DIM + threadIdx_x;
  yIndex = blockIdx_x * TILE_DIM + threadIdx_y;
  int index_out = xIndex + (yIndex)*height;

  for (int i=0; i<TILE_DIM; i += BLOCK_ROWS)
  {
    tile[(threadIdx_y+i)*(TILE_DIM+1)+threadIdx_x] = idata[index_in+i*width];
  }

  __syncthreads();

  for (int i=0; i<TILE_DIM; i += BLOCK_ROWS)
  {
    odata[index_out+i*height] = tile[(threadIdx_y+i)*(TILE_DIM+1)+threadIdx_x];
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

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  const char *kernelName;

  // execution configuration parameters
  sycl::range<2> gws (size_y/TILE_DIM * BLOCK_ROWS, size_x/TILE_DIM * TILE_DIM);
  sycl::range<2> lws (BLOCK_ROWS, TILE_DIM);

  if (size_y/TILE_DIM < 1 || size_x/TILE_DIM < 1)
  {
    printf("grid size computation incorrect in test \nExiting...\n\n");
    exit(EXIT_FAILURE);
  }

  // size of memory required to store the matrix
  size_t mat_size = size_x*size_y;
  size_t mem_size = static_cast<size_t>(sizeof(float) * mat_size);

  // allocate host memory
  float *h_idata = (float *) malloc(mem_size);
  float *h_odata = (float *) malloc(mem_size);
  float *transposeGold = (float *) malloc(mem_size);
  float *gold;

  // initialize host data
  for (int i = 0; i < (size_x*size_y); ++i)
  {
    h_idata[i] = (float) i;
  }

  // allocate device memory
  float *d_idata = sycl::malloc_device<float>(mat_size, q);
  q.memcpy(d_idata, h_idata, sizeof(float) * mat_size).wait();

  float *d_odata = sycl::malloc_device<float>(mat_size, q);

  // Compute reference transpose solution
  computeTransposeGold(transposeGold, h_idata, size_x, size_y);

  // print out common data for all kernels
  printf("\nMatrix size: %dx%d (%dx%d tiles), tile size: %dx%d, block size: %dx%d\n\n",
      size_x, size_y, size_x/TILE_DIM, size_y/TILE_DIM, TILE_DIM, TILE_DIM, TILE_DIM, BLOCK_ROWS);

  //
  // loop over different kernels
  //

  bool success = true;

  for (int k = 0; k < 8; k++)
  {
    switch (k)
    {
      case 0: {
        kernelName = "simple copy       ";
        auto start = std::chrono::steady_clock::now();

        for (int i = 0; i < repeat; i++) {
          q.submit([&] (sycl::handler &cgh) {
            cgh.parallel_for<class simple_copy>(
              sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
              copy(item, d_odata, d_idata, size_x, size_y);
            });
          });
        }

        q.wait();
        auto end = std::chrono::steady_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        printf("Average kernel (%s) execution time: %f (us)\n", kernelName, (time * 1e-3f) / repeat);
      }
      break;

      case 1: {
        kernelName = "shared memory copy";
        auto start = std::chrono::steady_clock::now();

        for (int i = 0; i < repeat; i++) {
          q.submit([&] (sycl::handler &cgh) {
            sycl::local_accessor<float, 1> sm (sycl::range<1>(TILE_DIM*TILE_DIM), cgh);
            cgh.parallel_for<class shared_mem_copy>(
              sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
              copySharedMem(
                item,
                sm.get_multi_ptr<sycl::access::decorated::no>().get(),
                d_odata,
                d_idata,
                size_x, size_y);
            });
          });
        }

        q.wait();
        auto end = std::chrono::steady_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        printf("Average kernel (%s) execution time: %f (us)\n", kernelName, (time * 1e-3f) / repeat);
      }
      break;

      case 2: {
        kernelName = "naive             ";
        auto start = std::chrono::steady_clock::now();

        for (int i = 0; i < repeat; i++) {
          q.submit([&] (sycl::handler &cgh) {
            cgh.parallel_for<class naive>(sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
              transposeNaive(
                item,
                d_odata,
                d_idata,
                size_x, size_y);
            });
          });
        }

        q.wait();
        auto end = std::chrono::steady_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        printf("Average kernel (%s) execution time: %f (us)\n", kernelName, (time * 1e-3f) / repeat);
      }
      break;

      case 3: {
        kernelName = "coalesced         ";
        auto start = std::chrono::steady_clock::now();

        for (int i = 0; i < repeat; i++) {
          q.submit([&] (sycl::handler &cgh) {
            sycl::local_accessor<float, 1> sm (sycl::range<1>(TILE_DIM*TILE_DIM), cgh);
            cgh.parallel_for<class coalesced>(
              sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
              transposeCoalesced(
                item,
                sm.get_multi_ptr<sycl::access::decorated::no>().get(),
                d_odata,
                d_idata,
                size_x, size_y);
            });
          });
        }

        q.wait();
        auto end = std::chrono::steady_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        printf("Average kernel (%s) execution time: %f (us)\n", kernelName, (time * 1e-3f) / repeat);
      }
      break;

      case 4: {
        kernelName = "optimized         ";
        auto start = std::chrono::steady_clock::now();

        for (int i = 0; i < repeat; i++) {
          q.submit([&] (sycl::handler &cgh) {
            sycl::local_accessor<float, 1> sm (sycl::range<1>(TILE_DIM*(TILE_DIM+1)), cgh);
            cgh.parallel_for<class optimized>(
              sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
              transposeNoBankConflicts(
                item,
                sm.get_multi_ptr<sycl::access::decorated::no>().get(),
                d_odata,
                d_idata,
                size_x, size_y);
            });
          });
        }

        q.wait();
        auto end = std::chrono::steady_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        printf("Average kernel (%s) execution time: %f (us)\n", kernelName, (time * 1e-3f) / repeat);
      }
      break;

      case 5: {
        kernelName = "coarse-grained    ";
        auto start = std::chrono::steady_clock::now();

        for (int i = 0; i < repeat; i++) {
          q.submit([&] (sycl::handler &cgh) {
            sycl::local_accessor<float, 1> sm (sycl::range<1>(TILE_DIM*(TILE_DIM+1)), cgh);
            cgh.parallel_for<class coarse_grained>(
              sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
              transposeCoarseGrained(
                item,
                sm.get_multi_ptr<sycl::access::decorated::no>().get(),
                d_odata,
                d_idata,
                size_x, size_y);
            });
          });
        }

        q.wait();
        auto end = std::chrono::steady_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        printf("Average kernel (%s) execution time: %f (us)\n", kernelName, (time * 1e-3f) / repeat);
      }
      break;

      case 6: {
        kernelName = "fine-grained      ";
        auto start = std::chrono::steady_clock::now();

        for (int i = 0; i < repeat; i++) {
          q.submit([&] (sycl::handler &cgh) {
            sycl::local_accessor<float, 1> sm (sycl::range<1>(TILE_DIM*(TILE_DIM+1)), cgh);
            cgh.parallel_for<class fine_grained>(
              sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
              transposeFineGrained(
                item,
                sm.get_multi_ptr<sycl::access::decorated::no>().get(),
                d_odata,
                d_idata,
                size_x, size_y);
            });
          });
        }

        q.wait();
        auto end = std::chrono::steady_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        printf("Average kernel (%s) execution time: %f (us)\n", kernelName, (time * 1e-3f) / repeat);
      }
      break;

      case 7: {
        kernelName = "diagonal          ";
        auto start = std::chrono::steady_clock::now();

        for (int i = 0; i < repeat; i++) {
          q.submit([&] (sycl::handler &cgh) {
            sycl::local_accessor<float, 1> sm (sycl::range<1>(TILE_DIM*(TILE_DIM+1)), cgh);
            cgh.parallel_for<class diagonal>(
              sycl::nd_range<2>(gws, lws), [=] (sycl::nd_item<2> item) {
              transposeDiagonal(
                item,
                sm.get_multi_ptr<sycl::access::decorated::no>().get(),
                d_odata,
                d_idata,
                size_x, size_y);
            });
          });
        }

        q.wait();
        auto end = std::chrono::steady_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        printf("Average kernel (%s) execution time: %f (us)\n", kernelName, (time * 1e-3f) / repeat);
      }
      break;
    }

    q.memcpy(h_odata, d_odata, mem_size).wait();

    // set reference solution
    if (k == 0 || k == 1)
    {
      gold = h_idata;
    }
    else if (k == 5 || k == 6)
    {
      gold = h_odata;   // fine- and coarse-grained kernels are not full transposes, so bypass check
    }
    else
    {
      gold = transposeGold;
    }

    bool ok = true;
    for (int i = 0; i < size_x*size_y; i++)
      if (fabsf(gold[i] - h_odata[i]) > 0.01f) {
        ok = false;
        break;
      }

    if (!ok)
    {
      printf("*** case %d: kernel FAILED ***\n", k);
      success = false;
    }
  }

  printf("%s\n", success ? "PASS" : "FAIL");
  if (!success) exit(1);

  // cleanup
  free(h_idata);
  free(h_odata);
  free(transposeGold);
  sycl::free(d_idata, q);
  sycl::free(d_odata, q);

  return 0;
}
