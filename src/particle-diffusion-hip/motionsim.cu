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
#include <string.h>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <hip/hip_runtime.h>
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

__global__
void Simulation(float*__restrict__ a_particleX,
                float*__restrict__ a_particleY,
		const float*__restrict__ a_randomX,
                const float*__restrict__ a_randomY, 
		size_t *__restrict__ a_map,
                const size_t n_particles,
                unsigned int nIterations,
                int grid_size,
                float radius)
{
  size_t ii = blockDim.x * blockIdx.x + threadIdx.x;
  if (ii >= n_particles) return;
  // Start iterations
  // Each iteration:
  //  1. Updates the position of all water molecules
  //  2. Checks if water molecule is inside a cell or not.
  //  3. Updates counter in cells array
  size_t iter = 0;
  float pX = a_particleX[ii];
  float pY = a_particleY[ii];
  size_t map_base = ii * grid_size * grid_size;
  while (iter < nIterations) {
    // Computes random displacement for each molecule
    // This example shows random distances between
    // -0.05 units and 0.05 units in both X and Y directions
    // Moves each water molecule by a random vector

    float randnumX = a_randomX[iter * n_particles + ii];
    float randnumY = a_randomY[iter * n_particles + ii];

    // Transform the scaled random numbers into small displacements
    float displacementX = randnumX / 1000.0f - 0.0495f;
    float displacementY = randnumY / 1000.0f - 0.0495f;

    // Move particles using random displacements
    pX += displacementX;
    pY += displacementY;

    // Compute distances from particle position to grid point
    float dX = pX - truncf(pX);
    float dY = pY - truncf(pY);

    // Compute grid point indices
    int iX = floorf(pX);
    int iY = floorf(pY);

    // Check if particle is still in computation grid
    if ((pX < grid_size) && (pY < grid_size) && (pX >= 0) && (pY >= 0)) {
      // Check if particle is (or remained) inside cell.
      // Increment cell counter in map array if so
      if ((dX * dX + dY * dY <= radius * radius))
        // The map array is organized as (particle, y, x)
        a_map[map_base + iY * grid_size + iX]++;
    }

    iter++;

  }  // Next iteration

  a_particleX[ii] = pX;
  a_particleY[ii] = pY;
}

// This function distributes simulation work across workers
void motion_device(float* particleX, float* particleY,
                   float* randomX, float* randomY, int** grid, size_t grid_size,
                   size_t n_particles, int nIterations, float radius,
                   size_t* map, int nRepeat) {

  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);

  std::cout << " Running on " << devProp.name << std::endl;
  std::cout << " The device max work-group size is " << devProp.maxThreadsPerBlock << std::endl;
  std::cout << " The number of iterations is " << nIterations << std::endl;
  std::cout << " The number of kernel execution is " << nRepeat << std::endl;
  std::cout << " The number of particles is " << n_particles << std::endl;
  
  float *d_randomX;
  float *d_randomY;
  float *d_particleX;
  float *d_particleY;
  size_t *d_map;
  hipMalloc((void**)&d_randomX, sizeof(float) * n_particles * nIterations);
  hipMalloc((void**)&d_randomY, sizeof(float) * n_particles * nIterations);
  hipMalloc((void**)&d_particleX, sizeof(float) * n_particles);
  hipMalloc((void**)&d_particleY, sizeof(float) * n_particles);
  size_t MAP_SIZE = n_particles * grid_size * grid_size;
  hipMalloc((void**)&d_map, sizeof(size_t) * MAP_SIZE);

  hipMemcpy(d_randomX, randomX, sizeof(float) * n_particles * nIterations, hipMemcpyHostToDevice);
  hipMemcpy(d_randomY, randomY, sizeof(float) * n_particles * nIterations, hipMemcpyHostToDevice);

  double time_total = 0.0;

  for (int i = 0; i < nRepeat; i++) {
    // reset the device data
    hipMemcpy(d_particleX, particleX, sizeof(float) * n_particles, hipMemcpyHostToDevice);
    hipMemcpy(d_particleY, particleY, sizeof(float) * n_particles, hipMemcpyHostToDevice);
    hipMemcpy(d_map, map, sizeof(size_t) * MAP_SIZE, hipMemcpyHostToDevice);

    hipDeviceSynchronize();
    auto start = std::chrono::steady_clock::now();

    Simulation<<< dim3((n_particles + 255) / 256), dim3(256) >>> (
      d_particleX, 
      d_particleY, 
      d_randomX, 
      d_randomY, 
      d_map, 
      n_particles,
      nIterations,
      grid_size,
      radius);

    hipDeviceSynchronize();
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    time_total += time;
  }

  std::cout << std::endl;
  std::cout << "Average kernel execution time: " << (time_total * 1e-9) / nRepeat << " (s)";
  std::cout << std::endl;

  hipMemcpy(map, d_map, sizeof(size_t) * MAP_SIZE, hipMemcpyDeviceToHost);

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

  hipFree(d_randomX);
  hipFree(d_randomY);
  hipFree(d_particleX);
  hipFree(d_particleY);
  hipFree(d_map);
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
  size_t MAP_SIZE = n_particles * grid_size * grid_size;
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
