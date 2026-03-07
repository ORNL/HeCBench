/***************************************************************************
Copyright 2020 Intel Corporation

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


 Description:
   This code sample will implement a simple example of a Monte Carlo
   simulation of the diffusion of water molecules in tissue.

**************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sycl/sycl.hpp>
#include "reference.h"

// Helper functions

// This function displays correct usage and parameters
void usage(std::string programName) {
  std::cout << " Incorrect number of parameters " << std::endl;
  std::cout << " Usage: ";
  std::cout << programName << " <Number of iterations within the kernel> ";
  std::cout << "<Kernel execution count>\n\n";
}

// This function prints a 2D matrix
template <typename T>
void print_matrix(T** matrix, size_t size_X, size_t size_Y) {
  std::cout << std::endl;
  for (size_t i = 0; i < size_X; ++i) {
    for (size_t j = 0; j < size_Y; ++j) {
      std::cout << std::setw(3) << matrix[i][j] << " ";
    }
    std::cout << std::endl;
  }
}

// This function prints a vector
template <typename T>
void print_vector(T* vector, size_t n) {
  std::cout << std::endl;
  for (size_t i = 0; i < n; ++i) {
    std::cout << vector[i] << " ";
  }
  std::cout << std::endl;
}

// This function distributes simulation work across workers
void motion_device(float* particleX, float* particleY,
                   float* randomX, float* randomY, int** grid, size_t grid_size,
                   size_t n_particles, int nIterations, float radius,
                   size_t* map, int nRepeat) {

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  auto device = q.get_device();
  auto deviceName = device.get_info<sycl::info::device::name>();
  auto maxBlockSize = device.get_info<sycl::info::device::max_work_group_size>();

  std::cout << " Running on " << deviceName << std::endl;
  std::cout << " The device max work-group size is " << maxBlockSize << std::endl;
  std::cout << " The number of iterations is " << nIterations << std::endl;
  std::cout << " The number of kernel execution is " << nRepeat << std::endl;
  std::cout << " The number of particles is " << n_particles << std::endl;

  const size_t map_size = n_particles * grid_size * grid_size;

  float *d_randomX = sycl::malloc_device<float>(n_particles * nIterations, q);
  q.memcpy(d_randomX, randomX, n_particles * nIterations * sizeof(float));

  float *d_randomY = sycl::malloc_device<float>(n_particles * nIterations, q);
  q.memcpy(d_randomY, randomY, n_particles * nIterations * sizeof(float));

  float *d_particleX = sycl::malloc_device<float>(n_particles, q);
  float *d_particleY = sycl::malloc_device<float>(n_particles, q);
  size_t *d_map = sycl::malloc_device<size_t>(map_size, q);

  sycl::range<1> gws ((n_particles + 255) / 256 * 256);
  sycl::range<1> lws (256);

  double time_total = 0.0;

  for (int i = 0; i < nRepeat; i++) {

    q.memcpy(d_particleX, particleX, n_particles * sizeof(float));
    q.memcpy(d_particleY, particleY, n_particles * sizeof(float));
    q.memcpy(d_map, map, map_size * sizeof(size_t));

    q.wait();
    auto start = std::chrono::steady_clock::now();

    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<class motionsim>(
        sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
        size_t ii = item.get_global_id(0);
        if (ii >= n_particles) return;

        // Start iterations
        // Each iteration:
        //  1. Updates the position of all water molecules
        //  2. Checks if water molecule is inside a cell or not.
        //  3. Updates counter in cells array
        float pX = d_particleX[ii];
        float pY = d_particleY[ii];
        size_t map_base = ii * grid_size * grid_size;
        size_t iter = 0;
        while (iter < nIterations) {
          // Computes random displacement for each molecule
          // This example shows random distances between
          // -0.05 units and 0.05 units in both X and Y directions
          // Moves each water molecule by a random vector

          float randnumX = d_randomX[iter * n_particles + ii];
          float randnumY = d_randomY[iter * n_particles + ii];

          // Transform the scaled random numbers into small displacements
          float displacementX = (float)randnumX / 1000.0f - 0.0495f;
          float displacementY = (float)randnumY / 1000.0f - 0.0495f;

          // Move particles using random displacements
          pX += displacementX;
          pY += displacementY;

          // Compute distances from particle position to grid point
          float dX = pX - sycl::trunc(pX);
          float dY = pY - sycl::trunc(pY);

          // Compute grid point indices
          int iX = sycl::floor(pX);
          int iY = sycl::floor(pY);

          // Check if particle is still in computation grid
          if ((pX < grid_size) & (pY < grid_size) & (pX >= 0) & (pY >= 0)) {
            // Check if particle is (or remained) inside cell.
            // Increment cell counter in map array if so
            if (dX * dX + dY * dY <= radius * radius)
              // The map array is organized as (particle, y, x)
              d_map[map_base + iY * grid_size + iX]++;
          }

          iter++;
        }  // Next iteration

        d_particleX[ii] = pX;
        d_particleY[ii] = pY;

      });
    }).wait();

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    time_total += time;
  }

  std::cout << std::endl;
  std::cout << "Average kernel execution time: " << (time_total * 1e-9) / nRepeat << " (s)";
  std::cout << std::endl;

  q.memcpy(map, d_map, map_size * sizeof(size_t)).wait();

  // For every cell in the grid, add all the counters from different
  // particles (workers) which are stored in the 3rd dimension of the 'map'
  // array
  for (size_t i = 0; i < n_particles; ++i) {
    for (size_t y = 0; y < grid_size; y++) {
      for (size_t x = 0; x < grid_size; x++) {
        if (map[i * grid_size * grid_size + y * grid_size + x] > 0) {
          grid[y][x] += map[i * grid_size * grid_size + y * grid_size + x];
        }
      }
    }
  }  // End loop for number of particles

  sycl::free(d_randomX, q);
  sycl::free(d_randomY, q);
  sycl::free(d_particleX, q);
  sycl::free(d_particleY, q);
  sycl::free(d_map, q);
}  // End of function motion_device()


int main(int argc, char* argv[]) {
  if (argc != 3) {
    usage(argv[0]);
    return 1;
  }

  // Read command-line arguments
  int nIterations = std::stoi(argv[1]);
  int nRepeat = std::stoi(argv[2]);

  // Cell and Particle parameters
  const size_t grid_size = 21;    // Size of square grid
  const size_t n_particles = 147456;  // Number of particles
  const float radius = 0.5;       // Cell radius = 0.5*(grid spacing)

  // Stores a grid of cells
  int** grid = new int*[grid_size];
  for (size_t i = 0; i < grid_size; i++) grid[i] = new int[grid_size];

  // Stores all random numbers to be used in the simulation
  float* randomX = new float[n_particles * nIterations];
  float* randomY = new float[n_particles * nIterations];

  // Stores X and Y position of particles in the cell grid
  float* particleX = new float[n_particles];
  float* particleY = new float[n_particles];

  // 'map' array replicates grid to be used by each particle
  const size_t MAP_SIZE = n_particles * grid_size * grid_size;
  size_t* map = new size_t[MAP_SIZE];
  size_t* map_ref = new size_t[MAP_SIZE];

  // Initialize arrays
  for (size_t i = 0; i < n_particles; i++) {
    // Initial position of particles in cell grid
    particleX[i] = 10.0;
    particleY[i] = 10.0;

    for (size_t y = 0; y < grid_size; y++) {
      for (size_t x = 0; x < grid_size; x++) {
        map[i * grid_size * grid_size + y * grid_size + x] = 0;
        map_ref[i * grid_size * grid_size + y * grid_size + x] = 0;
      }
    }
  }

  for (size_t y = 0; y < grid_size; y++) {
    for (size_t x = 0; x < grid_size; x++) {
      grid[y][x] = 0;
    }
  }


  // Compute vectors of random values for X and Y directions
  // Set the seed for rand() function.
  // Use a fixed seed for reproducibility/debugging
  srand(17);
  const size_t scale = 100; // Scale of random numbers

  for (size_t i = 0; i < n_particles * nIterations; i++) {
    randomX[i] = rand() % scale;
    randomY[i] = rand() % scale;
  }

  // Start timers
  auto start = std::chrono::steady_clock::now();

  // Call simulation function
  motion_device(particleX, particleY, randomX, randomY, grid, grid_size,
                n_particles, nIterations, radius, map, nRepeat);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  std::cout << std::endl;
  std::cout << "Simulation time: " << time * 1e-9 << " (s) ";
  std::cout << std::endl;

  motion_host(particleX, particleY, randomX, randomY, grid, grid_size,
              n_particles, nIterations, radius, map_ref, nRepeat);

  size_t count = 0;
  for (size_t i = 0; i < MAP_SIZE; i++) {
    if (map[i] != map_ref[i]) count++; 
  }
  std::cout << (count <= 2 ? "PASS" : "FAIL") << std::endl;
  if (count > 2) exit(1);

#ifdef DEBUG
  // Displays final grid only if grid small.
  if (grid_size <= 64) {
    std::cout << "\n ********************** OUTPUT GRID: " << std::endl;
    print_matrix<int>(grid, grid_size, grid_size);
  }
#endif

  // Cleanup
  for (size_t i = 0; i < grid_size; i++) delete grid[i];

  delete[] grid;
  delete[] particleX;
  delete[] particleY;
  delete[] randomX;
  delete[] randomY;
  delete[] map;
  delete[] map_ref;

  return 0;
}
