//==============================================================
// Copyright © 2019 Intel Corporation
//
// SPDX-License-Identifier: MIT
// =============================================================

// ISO2DFD: the 2D-Finite-Difference-Wave Propagation, 
//
// ISO2DFD is a finite difference stencil kernel for solving the 2D acoustic
// isotropic wave equation. Kernels in this sample are implemented as 2nd order
// in space, 2nd order in time scheme without boundary conditions.
// The sample will explicitly run on the GPU as well as CPU to
// calculate a result.  If successful, the output will include GPU device name.
//
// A complete online tutorial for this code sample can be found at :
// https://software.intel.com/en-us/articles/code-sample-two-dimensional-finite-difference-wave-propagation-in-isotropic-media-iso2dfd


#include <fstream>
#include <iostream>
#include <omp.h>
#include "iso2dfd.h"
#include <cstdlib>

#define MIN(a, b) (a) < (b) ? (a) : (b)

/*
 * Host-Code
 * Utility function to display input arguments
 */
void usage(std::string programName) {
  std::cout << " Incorrect parameters " << std::endl;
  std::cout << " Usage: ";
  std::cout << programName << " n1 n2 Iterations " << std::endl
            << std::endl;
  std::cout << " n1 n2      : Grid sizes for the stencil " << std::endl;
  std::cout << " Iterations : No. of timesteps. " << std::endl;
}

/*
 * Host-Code
 * Function used for initialization
 */
void initialize(float* ptr_prev, float* ptr_next, float* ptr_vel, size_t nRows,
                size_t nCols) {
  std::cout << "Initializing ... " << std::endl;

  // Define source wavelet
  float wavelet[12] = {0.016387336, -0.041464937, -0.067372555, 0.386110067,
                       0.812723635, 0.416998396,  0.076488599,  -0.059434419,
                       0.023680172, 0.005611435,  0.001823209,  -0.000720549};

  // Initialize arrays
  for (size_t i = 0; i < nRows; i++) {
    size_t offset = i * nCols;

    for (int k = 0; k < nCols; k++) {
      ptr_prev[offset + k] = 0.0f;
      ptr_next[offset + k] = 0.0f;
      // pre-compute squared value of sample wave velocity v*v (v = 1500 m/s)
      ptr_vel[offset + k] = 2250000.0f;
    }
  }
  // Add a source to initial wavefield as an initial condition
  for (int s = 11; s >= 0; s--) {
    for (size_t i = nRows / 2 - s; i < nRows / 2 + s; i++) {
      size_t offset = i * nCols;
      for (size_t k = nCols / 2 - s; k < nCols / 2 + s; k++) {
        ptr_prev[offset + k] = wavelet[s];
      }
    }
  }
}

/*
 * Host-Code
 * Utility function to calculate L2-norm between resulting buffer and reference
 * buffer
 */
bool within_epsilon(float* output, float* reference, const size_t dimx,
                    const size_t dimy, const unsigned int radius,
                    const float delta = 0.01f) {
  FILE* fp = fopen("./error_diff.txt", "w");
  if (!fp) fp = stderr;

  bool error = false;
  //float abs_delta = fabsf(delta);
  double norm2 = 0;

  for (size_t iy = 0; iy < dimy; iy++) {
    for (size_t ix = 0; ix < dimx; ix++) {
      if (ix >= radius && ix < (dimx - radius) && iy >= radius &&
          iy < (dimy - radius)) {
        float difference = fabsf(*reference - *output);
        norm2 += difference * difference;
        if (difference > delta) {
          error = true;
          fprintf(fp, " ERROR: (%zu,%zu)\t%e instead of %e (|e|=%e)\n", ix, iy,
                  *output, *reference, difference);
        }
      }

      ++output;
      ++reference;
    }
  }

  if (fp != stderr) fclose(fp);
  norm2 = sqrt(norm2);
  if (error) printf("error (Euclidean norm): %.9e\n", norm2);
  return error;
}

/*
 * Host-Code
 * CPU implementation for wavefield modeling
 * Updates wavefield for the number of iterations given in nIteratons parameter
 */
void iso_2dfd_iteration_cpu(float* next, float* prev, float* vel,
                            const float dtDIVdxy, int nRows, int nCols,
                            int nIterations) {
  for (unsigned int k = 0; k < nIterations; k += 1) {
    for (size_t i = 1; i < nRows - HALF_LENGTH; i += 1) {
      for (size_t j = 1; j < nCols - HALF_LENGTH; j += 1) {
        // Stencil code to update grid
        size_t gid = j + (i * nCols);
        float value = 0.f;
        value += prev[gid + 1] - 2.f * prev[gid] + prev[gid - 1];
        value += prev[gid + nCols] - 2.f * prev[gid] + prev[gid - nCols];
        value *= dtDIVdxy * vel[gid];
        next[gid] = 2.f * prev[gid] - next[gid] + value;
      }
    }

    // Swap arrays
    float* swap = next;
    next = prev;
    prev = swap;
  }
}

/*
 * Device-Code - GPU
 * SYCL implementation for single iteration of iso2dfd kernel
 *
 * Range kernel is used to spawn work-items in x, y dimension
 *
 */
void iso_2dfd_kernel(float* next, const float* prev, const float* vel, 
                     const float dtDIVdxy, const size_t nRows, const size_t nCols) {
  #pragma omp target teams distribute parallel for simd collapse(2) thread_limit(256) 
  for (size_t gidRow = 0; gidRow < nRows ; gidRow++)
    for (size_t gidCol = 0; gidCol < nCols ; gidCol++) {
      size_t gid = (gidRow)*nCols + gidCol;
      // Computation to solve wave equation in 2D
      // First check if gid is inside the effective grid (not in halo)
      if ((gidCol >= HALF_LENGTH && gidCol < nCols - HALF_LENGTH) &&
          (gidRow >= HALF_LENGTH && gidRow < nRows - HALF_LENGTH)) {
        // Stencil code to update grid point at position given by global id (gid)
        // New time step for grid point is computed based on the values of the
        //    the immediate neighbors in both the horizontal and vertical
        //    directions, as well as the value of grid point at a previous time step
        float value = 0.f;
        value += prev[gid + 1] - 2.f * prev[gid] + prev[gid - 1];
        value += prev[gid + nCols] - 2.f * prev[gid] + prev[gid - nCols];
        value *= dtDIVdxy * vel[gid];
        next[gid] = 2.f * prev[gid] - next[gid] + value;
      }
  }
}

int main(int argc, char* argv[]) {
  // Arrays used to update the wavefield
  float* prev_base;
  float* next_base;
  float* next_cpu;
  // Array to store wave velocity
  float* vel_base;

  bool error = false;

  size_t nRows, nCols;
  unsigned int nIterations;

  // Read parameters
  try {
    nRows = std::stoi(argv[1]);
    nCols = std::stoi(argv[2]);
    nIterations = std::stoi(argv[3]);
  }

  catch (...) {
    usage(argv[0]);
    return 1;
  }

  // Compute the total size of grid
  size_t nsize = nRows * nCols;

  // Allocate arrays to hold wavefield and velocity
  prev_base = new float[nsize];
  next_base = new float[nsize];
  next_cpu = new float[nsize];
  vel_base = new float[nsize];

  // Compute constant value (delta t)^2 (delta x)^2. To be used in wavefield
  // update
  float dtDIVdxy = (DT * DT) / (DXY * DXY);

  // Initialize arrays and introduce initial conditions (source)
  initialize(prev_base, next_base, vel_base, nRows, nCols);

  std::cout << "Grid Sizes: " << nRows << " " << nCols << std::endl;
  std::cout << "Iterations: " << nIterations << std::endl;
  std::cout << std::endl;

  std::cout << "Computing wavefield in device .." << std::endl;

  #pragma omp target data map(next_base[0:nsize], prev_base[0:nsize]) \
                          map(to: vel_base[0:nsize])
  {
    auto kstart = std::chrono::steady_clock::now();
  
    // Iterate over time steps
    for (unsigned int k = 0; k < nIterations; k += 1) {
      // alternating the 'next' and 'prev' parameters which effectively
      //    swaps their content at every iteration.
      iso_2dfd_kernel((k % 2) ? prev_base : next_base,
                      (k % 2) ? next_base : prev_base,
                      vel_base, dtDIVdxy, nRows, nCols);
    }  // end for
  
    auto kend = std::chrono::steady_clock::now();
    auto ktime = std::chrono::duration_cast<std::chrono::nanoseconds>(kend - kstart).count();
    std::cout << "Total kernel execution time " << ktime * 1e-6f << " (ms)\n";
    std::cout << "Average kernel execution time " << (ktime * 1e-3f) / nIterations << " (us)\n";
  }

  // Output final wavefield (computed by device) to binary file
  std::ofstream outFile;
  outFile.open("wavefield_snapshot.bin", std::ios::out | std::ios::binary);
  outFile.write(reinterpret_cast<char*>(next_base), nsize * sizeof(float));
  outFile.close();

  // Compute wavefield on CPU (for validation)
  
  std::cout << "Computing wavefield in CPU .." << std::endl;
  // Re-initialize arrays
  initialize(prev_base, next_cpu, vel_base, nRows, nCols);

  // Compute wavefield on CPU
  // Start timer for CPU
  auto start = std::chrono::steady_clock::now();
  iso_2dfd_iteration_cpu(next_cpu, prev_base, vel_base, dtDIVdxy, nRows, nCols,
                         nIterations);

  // Compute and display time used by CPU
  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                  .count();
  std::cout << "CPU time: " << time << " ms" << std::endl;
  std::cout << std::endl;

  std::cout << "Check difference between final wavefields computed in device and host"
            << std::endl;
  error = within_epsilon(next_base, next_cpu, nRows, nCols, HALF_LENGTH, 0.1f);
  std::cout << (error ? "FAIL" : "PASS") << std::endl;
  if (error) exit(1);

  // Output final wavefield (computed by CPU) to binary file
  outFile.open("wavefield_snapshot_cpu.bin", std::ios::out | std::ios::binary);
  outFile.write(reinterpret_cast<char*>(next_cpu), nsize * sizeof(float));
  outFile.close();

  std::cout << "Final wavefields (from device and CPU) written to disk"
            << std::endl;
  std::cout << "Finished.  " << std::endl;

  // Cleanup
  delete[] prev_base;
  delete[] next_base;
  delete[] vel_base;

  return error ? 1 : 0;
}
