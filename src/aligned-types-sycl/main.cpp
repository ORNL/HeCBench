/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This is a simple test showing performance differences
 * between aligned and misaligned structures
 * (those having/missing __attribute__((__aligned__ keyword).
 * It measures per-element copy throughput for
 * aligned and misaligned structures on
 * big chunks of data.
 */


#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <chrono>
#include <sycl/sycl.hpp>


// Forward declaration
template <typename TData>
class copy_kernel;

////////////////////////////////////////////////////////////////////////////////
// Misaligned types
////////////////////////////////////////////////////////////////////////////////
typedef unsigned char uchar_misaligned;

typedef unsigned short int ushort_misaligned;

struct uchar4_misaligned
{
  unsigned char r, g, b, a;
};

struct uint2_misaligned
{
  unsigned int l, a;
};

struct uint3_misaligned
{
  unsigned int r, g, b;
};

struct uint4_misaligned
{
  unsigned int r, g, b, a;
};

struct uint8_misaligned
{
  uint4_misaligned c1, c2;
};

////////////////////////////////////////////////////////////////////////////////
// Aligned types
////////////////////////////////////////////////////////////////////////////////
struct alignas(4) uchar4_aligned
{
  unsigned char r, g, b, a;
};

typedef unsigned int uint_aligned;

struct alignas(8) uint2_aligned
{
  unsigned int l, a;
};

struct alignas(16) uint3_aligned
{
  unsigned int r, g, b;
};

struct alignas(16) uint4_aligned
{
  unsigned int r, g, b, a;
};


////////////////////////////////////////////////////////////////////////////////
// Because G80 class hardware natively supports global memory operations
// only with data elements of 4, 8 and 16 bytes, if structure size
// exceeds 16 bytes, it can't be efficiently read or written,
// since more than one global memory non-coalescable load/store instructions
// will be generated, even if __attribute__((__aligned__ option is supplied.
// "Structure of arrays" storage strategy offers best performance
// in general case. See section 5.1.2 of the Programming Guide.
////////////////////////////////////////////////////////////////////////////////
struct alignas(16) uint8_aligned
{
  uint4_aligned c1, c2;
};



////////////////////////////////////////////////////////////////////////////////
// Common host and device functions
////////////////////////////////////////////////////////////////////////////////
//Round a / b to nearest higher integer value
int iDivUp(int a, int b)
{
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Round a / b to nearest lower integer value
int iDivDown(int a, int b)
{
  return a / b;
}

//Align a to nearest higher multiple of b
int iAlignUp(int a, int b)
{
  return (a % b != 0) ? (a - a % b + b) : a;
}

//Align a to nearest lower multiple of b
int iAlignDown(int a, int b)
{
  return a - a % b;
}



////////////////////////////////////////////////////////////////////////////////
// Copy is carried out on per-element basis,
// so it's not per-byte in case of padded structures.
////////////////////////////////////////////////////////////////////////////////
template <typename TData> 
void testKernel(      TData *__restrict d_odata,
                const TData *__restrict d_idata,
                int numElements,
                sycl::nd_item<1> &item)
{
  const int pos = item.get_global_id(0);
  if (pos < numElements)
  {
    d_odata[pos] = d_idata[pos];
  }
}



////////////////////////////////////////////////////////////////////////////////
// Validation routine for simple copy kernel.
// We must know "packed" size of TData (number_of_fields * sizeof(simple_type))
// and compare only these "packed" parts of the structure,
// containing actual user data. The compiler behavior with padding bytes
// is undefined, since padding is merely a placeholder
// and doesn't contain any user data.
////////////////////////////////////////////////////////////////////////////////
template <typename TData> 
int testCPU(
    TData *h_odata,
    TData *h_idata,
    int numElements,
    int packedElementSize)
{
  for (int pos = 0; pos < numElements; pos++)
  {
    TData src = h_idata[pos];
    TData dst = h_odata[pos];

    for (int i = 0; i < packedElementSize; i++)
      if (((char *)&src)[i] != ((char *)&dst)[i])
      {
        return 0;
      }
  }
  return 1;
}



////////////////////////////////////////////////////////////////////////////////
// Data configuration
////////////////////////////////////////////////////////////////////////////////
//Memory chunk size in bytes. Reused for test
const int       MEM_SIZE = 50000000;
const int NUM_ITERATIONS = 1000;

//GPU input and output data
unsigned char *d_idata, *d_odata;
//CPU input data and instance of GPU output data
unsigned char *h_idataCPU, *h_odataGPU;

template <typename TData> 
int runTest(
  sycl::queue &q,
  unsigned char *d_idata,
  unsigned char *d_odata,
  int packedElementSize,
  int memory_size)
{
  const int totalMemSizeAligned = iAlignDown(memory_size, sizeof(TData));
  const int         numElements = iDivDown(memory_size, sizeof(TData));

  //Clean output buffer before current test
  q.memset(d_odata, 0, memory_size).wait();

  //Run test
  sycl::range<1> gws ((numElements + 255)/256*256);
  sycl::range<1> lws (256);

  auto start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < NUM_ITERATIONS; i++)
  {
    q.submit([&] (sycl::handler &cgh) {
      cgh.parallel_for<class copy_kernel<TData>>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        testKernel<TData>((TData*)d_odata, (TData*)d_idata, numElements, item);
      });
    });
  }

  q.wait();

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  double gpuTime = (double)elapsed_seconds.count() / NUM_ITERATIONS;

  printf("Avg. time: %f ms / Copy throughput: %f GB/s.\n", gpuTime * 1000,
         (double)totalMemSizeAligned / (gpuTime * 1073741824.0));

  //Read back GPU results and run validation
  q.memcpy(h_odataGPU, d_odata, memory_size).wait();

  int flag = testCPU((TData *)h_odataGPU,
                     (TData *)h_idataCPU,
                     numElements,
                     packedElementSize);

  printf("\tTEST %s\n", flag ? "PASS" : "FAIL");
  if (!flag) exit(1);

  return !flag;
}

int main(int argc, char **argv)
{
  int i, nTotalFailures = 0;

  printf("[%s] - Starting...\n", argv[0]);

  printf("Allocating memory...\n");
  int   MemorySize = (int)(MEM_SIZE) & 0xffffff00; // force multiple of 256 bytes
  h_idataCPU = (unsigned char *)malloc(MemorySize);
  h_odataGPU = (unsigned char *)malloc(MemorySize);

  printf("Generating host input data array...\n");

  for (i = 0; i < MemorySize; i++)
  {
    h_idataCPU[i] = (i & 0xFF) + 1;
  }

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  printf("Uploading input data to GPU memory...\n");
  unsigned char *d_idata = sycl::malloc_device<unsigned char>(MemorySize, q);
  q.memcpy(d_idata, h_idataCPU, MemorySize);

  unsigned char *d_odata = sycl::malloc_device<unsigned char>(MemorySize, q);

  printf("Testing misaligned types...\n");
  printf("uchar_misaligned...\n");
  nTotalFailures += runTest<uchar_misaligned>(q, d_idata, d_odata, 1, MemorySize);

  printf("uchar4_misaligned...\n");
  nTotalFailures += runTest<uchar4_misaligned>(q, d_idata, d_odata, 4, MemorySize);

  printf("uchar4_aligned...\n");
  nTotalFailures += runTest<uchar4_aligned>(q, d_idata, d_odata, 4, MemorySize);

  printf("ushort_misaligned...\n");
  nTotalFailures += runTest<ushort_misaligned>(q, d_idata, d_odata, 2, MemorySize);

  printf("uint_aligned...\n");
  nTotalFailures += runTest<uint_aligned>(q, d_idata, d_odata, 4, MemorySize);

  printf("uint2_misaligned...\n");
  nTotalFailures += runTest<uint2_misaligned>(q, d_idata, d_odata, 8, MemorySize);

  printf("uint2_aligned...\n");
  nTotalFailures += runTest<uint2_aligned>(q, d_idata, d_odata, 8, MemorySize);

  printf("uint3_misaligned...\n");
  nTotalFailures += runTest<uint3_misaligned>(q, d_idata, d_odata, 12, MemorySize);

  printf("uint3_aligned...\n");
  nTotalFailures += runTest<uint3_aligned>(q, d_idata, d_odata, 12, MemorySize);

  printf("uint4_misaligned...\n");
  nTotalFailures += runTest<uint4_misaligned>(q, d_idata, d_odata, 16, MemorySize);

  printf("uint4_aligned...\n");
  nTotalFailures += runTest<uint4_aligned>(q, d_idata, d_odata, 16, MemorySize);

  printf("uint8_misaligned...\n");
  nTotalFailures += runTest<uint8_misaligned>(q, d_idata, d_odata, 32, MemorySize);

  printf("uint8_aligned...\n");
  nTotalFailures += runTest<uint8_aligned>(q, d_idata, d_odata, 32, MemorySize);

  printf("\n[alignedTypes] -> Test Results: %d Failures\n", nTotalFailures);

  printf("Shutting down...\n");
  sycl::free(d_odata, q);
  sycl::free(d_idata, q);
  free(h_odataGPU);
  free(h_idataCPU);

  if (nTotalFailures != 0)
  {
    printf("Test failed!\n");
    exit(EXIT_FAILURE);
  }

  printf("Test passed\n");
  exit(EXIT_SUCCESS);
}
