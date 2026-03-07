#include <math.h>
#include <string.h>
#include <sycl/sycl.hpp>
#include "OptionParser.h"
#include "Timer.h"
#include "Utility.h"
#include <cstdlib>

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
//   Implements the Stream Triad benchmark in CUDA.  This benchmark
//   is designed to test CUDA's overall data transfer speed. It executes
//   a vector addition operation with no temporal reuse. Data is read
//   directly from the global memory. This implementation tiles the input
//   array and pipelines the vector addition computation with
//   the data download for the next tile. However, since data transfer from
//   host to device is much more expensive than the simple vector computation,
//   data transfer operations should completely dominate the execution time.
//
// Arguments:
//   resultDB: results from the benchmark are stored in this db
//   op: the options parser (contains input parameters)
//
// Returns:  nothing
//
// Programmer: Kyle Spafford
// Creation: December 15, 2009
//
// Modifications:
//
// ****************************************************************************
void RunBenchmark(OptionParser &op)
{
  const bool verbose = op.getOptionBool("verbose");
  const int n_passes = op.getOptionInt("passes");

  const int nSizes = 9;
  const size_t blockSizes[] = { 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384 };
  const size_t memSize = 16384;
  const size_t numMaxFloats = 1024 * memSize / sizeof(float);
  const size_t halfNumFloats = numMaxFloats / 2;

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  // Create some host memory pattern
  srand48(8650341L);
  float *h_mem = sycl::malloc_host<float> (numMaxFloats, q);

  // Allocate device memory of maximum sizes
  float *d_memA0 = sycl::malloc_device<float>(numMaxFloats, q);
  float *d_memB0 = sycl::malloc_device<float>(numMaxFloats, q);
  float *d_memC0 = sycl::malloc_device<float>(numMaxFloats, q);
  float *d_memA1 = sycl::malloc_device<float>(numMaxFloats, q);
  float *d_memB1 = sycl::malloc_device<float>(numMaxFloats, q);
  float *d_memC1 = sycl::malloc_device<float>(numMaxFloats, q);

  const float scalar = 1.75f;
  const size_t blockSize = 128;

  // Step through sizes forward
  for (int i = 0; i < nSizes; ++i)
  {
    int elemsInBlock = blockSizes[i] * 1024 / sizeof(float);
    for (int j = 0; j < halfNumFloats; ++j)
      h_mem[j] = h_mem[halfNumFloats + j] = (float) (drand48() * 10.0);

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
    sycl::range<1> gws (elemsInBlock);
    sycl::range<1> lws (blockSize);

    int TH = Timer::Start();

    // Number of passes. Use a large number for stress testing.
    // A small value is sufficient for computing sustained performance.
    for (int pass = 0; pass < n_passes; ++pass)
    {
      q.memcpy(d_memA0, h_mem, sizeof(float) * elemsInBlock);
      q.memcpy(d_memB0, h_mem, sizeof(float) * elemsInBlock);

      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class triad_start>(
          sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
          int gid = item.get_global_id(0);
          d_memC0[gid] = d_memA0[gid] + scalar*d_memB0[gid];
        });
      });

      if (elemsInBlock < numMaxFloats)
      {
        // start downloading data for next block
        q.memcpy(d_memA1, h_mem + elemsInBlock, sizeof(float) * elemsInBlock);
        q.memcpy(d_memB1, h_mem + elemsInBlock, sizeof(float) * elemsInBlock);
      }

      int blockIdx = 1;
      unsigned int currStream = 1;
      while (crtIdx < numMaxFloats)
      {
        currStream = blockIdx & 1;
        // Start copying back the answer from the last kernel
        if (currStream)
        {
          q.memcpy(h_mem + crtIdx, d_memC0, sizeof(float) * elemsInBlock);
        }
        else
        {
          q.memcpy(h_mem + crtIdx, d_memC1, sizeof(float) * elemsInBlock);
        }

        crtIdx += elemsInBlock;

        if (crtIdx < numMaxFloats)
        {
          // Execute the kernel
          if (currStream)
          {
            q.submit([&] (sycl::handler &cgh) {
              cgh.parallel_for<class triad_curr>(
                sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
                int gid = item.get_global_id(0);
                d_memC1[gid] = d_memA1[gid] + scalar*d_memB1[gid];
              });
            });
          }
          else
          {
            q.submit([&] (sycl::handler &cgh) {
              cgh.parallel_for<class triad_next>(
                sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
                int gid = item.get_global_id(0);
                d_memC0[gid] = d_memA0[gid] + scalar*d_memB0[gid];
              });
            });
          }
        }

        if (crtIdx+elemsInBlock < numMaxFloats)
        {
          // Download data for next block
          if (currStream)
          {
            q.memcpy(d_memA0, h_mem+crtIdx+elemsInBlock, sizeof(float) * elemsInBlock);
            q.memcpy(d_memB0, h_mem+crtIdx+elemsInBlock, sizeof(float) * elemsInBlock);
          }
          else
          {
            q.memcpy(d_memA1, h_mem+crtIdx+elemsInBlock, sizeof(float) * elemsInBlock);
            q.memcpy(d_memB1, h_mem+crtIdx+elemsInBlock, sizeof(float) * elemsInBlock);
          }
        }
        blockIdx += 1;
        currStream = !currStream;
      }
      q.wait();
    }

    double time = Timer::Stop(TH, "thread synchronize");

    double triad = ((double)numMaxFloats*2.0*n_passes) / (time*1e9);
    if (verbose) std::cout << "Average TriadFlops " << triad << " GFLOPS/s\n";

    double bdwth = ((double)numMaxFloats*sizeof(float)*3.0*n_passes)
                   / (time*1000.*1000.*1000.);
    if (verbose) std::cout << "Average TriadBdwth " << bdwth << " GB/s\n";

    // Checking memory for correctness. The two halves of the array
    // should have the same results.
    bool ok = true;
    for (int j=0; j<halfNumFloats; ++j)
    {
      if (h_mem[j] != h_mem[j+halfNumFloats])
      {
        std::cout << "Error; hostMem[" << j << "]=" << h_mem[j]
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

    // Zero out the test host memory
    for (int j=0; j<numMaxFloats; ++j) h_mem[j] = 0.0f;
  }

  // Cleanup
  free(d_memA0, q);
  free(d_memB0, q);
  free(d_memC0, q);
  free(d_memA1, q);
  free(d_memB1, q);
  free(d_memC1, q);
  free(h_mem, q);
}
