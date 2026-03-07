#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "OptionParser.h"
#include "Timer.h"
#include "Utility.h"

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing
//
// Arguments:
//   op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: December 15, 2009
//
// Modifications:
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op)
{
  ;
}
// ****************************************************************************
// Function: RunBenchmark
//
// Purpose:
//   Implements the Stream Triad benchmark in OpenMP. This benchmark
//   is designed to test overall data transfer speed. It executes
//   a vector addition operation with no temporal reuse. Data is read
//   directly from the global memory. This implementation tiles the input
//   array and pipelines the vector addition computation with
//   the data download for the next tile. However, since data transfer from
//   host to device is much more expensive than the simple vector computation,
//   data transfer operations should completely dominate the execution time.
//
// Arguments:
//   op: the options parser (contains input parameters)
//
// ****************************************************************************
void RunBenchmark(OptionParser &op)
{
  const bool verbose = op.getOptionBool("verbose");
  const int n_passes = op.getOptionInt("passes");

  const int nSizes = 9;
  const int blockSizes[] = { 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384 };
  const int memSize = 16384;
  const int numMaxFloats = 1024 * memSize / sizeof(float);
  const int halfNumFloats = numMaxFloats / 2;
  const int maxBlockSize = blockSizes[nSizes - 1] * 1024;

  // Create some host memory pattern
  srand48(8650341L);
  float *h_mem = (float*) malloc (sizeof(float) * numMaxFloats);

  // Allocate device memory of maximum sizes
  float *A0 = (float*) malloc (sizeof(float) * maxBlockSize);
  float *B0 = (float*) malloc (sizeof(float) * maxBlockSize);
  float *C0 = (float*) malloc (sizeof(float) * maxBlockSize);
  float *A1 = (float*) malloc (sizeof(float) * maxBlockSize);
  float *B1 = (float*) malloc (sizeof(float) * maxBlockSize);
  float *C1 = (float*) malloc (sizeof(float) * maxBlockSize);

  const float scalar = 1.75f;
  const int blockSize = 128;

  #pragma omp target data map(alloc: A0[0:maxBlockSize],\
                                     B0[0:maxBlockSize],\
                                     C0[0:maxBlockSize],\
                                     A1[0:maxBlockSize],\
                                     B1[0:maxBlockSize],\
                                     C1[0:maxBlockSize])
  {
    // Step through sizes forward
    for (int i = 0; i < nSizes; ++i)
    {
      // Zero out the host memory
      for (int j=0; j<numMaxFloats; ++j)
        C0[j] = C1[j] = 0.0f;

      for (int j = 0; j < halfNumFloats; ++j) {
        A0[j] = A0[halfNumFloats + j] = B0[j] = B0[halfNumFloats + j] = \
        A1[j] = A1[halfNumFloats + j] = B1[j] = B1[halfNumFloats + j] \
              = (float) (drand48() * 10.0);
      }

      int elemsInBlock = blockSizes[i] * 1024 / sizeof(float);
      // Copy input memory to the device
      if (verbose) {
        std::cout << ">> Executing Triad with vectors of length "
          << numMaxFloats << " and block size of "
          << elemsInBlock << " elements." << "\n";
        std::cout << "Block: " << blockSizes[i] << "KB" << "\n";
      }

      // start submitting blocks of data of size elemsInBlock
      // overlap the computation of one block with the data
      // download for the next block and the results upload for
      // the previous block
      int crtIdx = 0;

      int TH = Timer::Start();

      // Number of passes. Use a large number for stress testing.
      // A small value is sufficient for computing sustained performance.
      for (int pass = 0; pass < n_passes; ++pass)
      {
        #pragma omp target update to (A0[0:elemsInBlock]) nowait
        #pragma omp target update to (B0[0:elemsInBlock]) nowait
        
        #pragma omp target teams distribute parallel for thread_limit(blockSize) nowait
        for (int gid = 0; gid < elemsInBlock; gid++) 
          C0[gid] = A0[gid] + scalar*B0[gid];

        if (elemsInBlock < numMaxFloats)
        {
          // start downloading data for next block
          #pragma omp target update to (A1[elemsInBlock:2*elemsInBlock]) nowait
          #pragma omp target update to (B1[elemsInBlock:2*elemsInBlock]) nowait
        }

        int blockIdx = 1;
        unsigned int currStream = 1;
        while (crtIdx < numMaxFloats)
        {
          currStream = blockIdx & 1;
          // Start copying back the answer from the last kernel
          if (currStream)
          {
            #pragma omp target update from(C0[crtIdx:crtIdx+elemsInBlock]) nowait
          }
          else
          {
            #pragma omp target update from(C1[crtIdx:crtIdx+elemsInBlock]) nowait
          }

          crtIdx += elemsInBlock;

          if (crtIdx < numMaxFloats)
          {
            // Execute the kernel
            if (currStream)
            {
              #pragma omp target teams distribute parallel for thread_limit(blockSize) nowait
              for (int gid = 0; gid < elemsInBlock; gid++) 
                C1[crtIdx+gid] = A1[crtIdx+gid] + scalar*B1[crtIdx+gid];
            }
            else
            {
              #pragma omp target teams distribute parallel for thread_limit(blockSize) nowait
              for (int gid = 0; gid < elemsInBlock; gid++) 
                C0[crtIdx+gid] = A0[crtIdx+gid] + scalar*B0[crtIdx+gid];
            }
          }

          if (crtIdx+elemsInBlock < numMaxFloats)
          {
            // Download data for next block
            if (currStream)
            {
              #pragma omp target update to (A0[crtIdx+elemsInBlock:crtIdx+2*elemsInBlock]) nowait
              #pragma omp target update to (B0[crtIdx+elemsInBlock:crtIdx+2*elemsInBlock]) nowait
            }
            else
            {
              #pragma omp target update to (A1[crtIdx+elemsInBlock:crtIdx+2*elemsInBlock]) nowait
              #pragma omp target update to (B1[crtIdx+elemsInBlock:crtIdx+2*elemsInBlock]) nowait
            }
          }
          blockIdx += 1;
          currStream = !currStream;
        }
      } // for (int pass = 0; pass < n_passes; ++pass)

      double time = Timer::Stop(TH, "Warning: no thread synchronization");

      double triad = ((double)numMaxFloats*2.0*n_passes) / (time*1e9);
      if (verbose) std::cout << "Average TriadFlops " << triad << " GFLOPS/s\n";

    double bdwth = ((double)numMaxFloats*sizeof(float)*3.0*n_passes)
                     / (time*1000.*1000.*1000.);
      if (verbose) std::cout << "Average TriadBdwth " << bdwth << " GB/s\n";

      // Checking memory for correctness. The two halves of the array
      // should have the same results.
      bool ok = true;
      for (int j=0; j<numMaxFloats; j=j+elemsInBlock) {
        if (((j / elemsInBlock) & 1) == 0) {
          memcpy(h_mem+j, C0+j, elemsInBlock*sizeof(float));
        }
        else {
          memcpy(h_mem+j, C1+j, elemsInBlock*sizeof(float));
        }
      }

      for (int j=0; j<halfNumFloats; ++j)
      {
        if (h_mem[j] != h_mem[j+halfNumFloats])
        {
          std::cout << "hostMem[" << j << "]=" << h_mem[j]
            << " is different from its twin element hostMem["
            << (j+halfNumFloats) << "]: "
            << h_mem[j+halfNumFloats] << "stopping check\n";
          ok = false;
          break;
        }
      }

      if (ok)
        std::cout << "PASS\n";
      else
        std::cout << "FAIL\n";
      if (!ok) exit(1);
    }
  }

  // Cleanup
  free(h_mem);
  free(A0);
  free(B0);
  free(C0);
  free(A1);
  free(B1);
  free(C1);
}
