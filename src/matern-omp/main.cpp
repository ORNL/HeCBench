#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include "reference.h"

//
// Assumption 
// There are many more evaluation(target) points than sources for the subsequent code. 
// Each thread block will perform the evaluation for a small chunk of the target points and all source points. 
// 
void matern_kernel (
  const int ntargets,
  const float l,
  const float *__restrict sources,
  const float *__restrict targets,
  const float *__restrict weights,
        float *__restrict result)

{
  #pragma omp target teams distribute thread_limit(SX*64)
  for (int t = 0; t < ntargets; t++) {
    float sum = 0.f;
    #pragma omp parallel for reduction(+:sum)
    for (int s = 0; s < nsources; s++) {
      float squared_diff = 0.f;
      for (int i = 0; i < 3; i++) {
        squared_diff += (sources[s*3+i] - targets[t*3+i]) *
                        (sources[s*3+i] - targets[t*3+i]);
      }
      float diff = sqrtf(squared_diff);
      sum += (1.f + sqrtf(5.f) * diff / l + 5.f * squared_diff / (3.f * l * l)) *  
             expf(-sqrtf(5.f) * diff  / l) * weights[s];
    }
    result[t] = sum;
  }
  #pragma omp target update from(result[0:ntargets])
}

void matern_kernel2 (
  const int ntargets,
  const float l,
  const float *__restrict sources,
  const float *__restrict targets,
  const float *__restrict weights,
        float *__restrict result)

{
  const int teams = (ntargets + SX - 1) / SX;

  // SY is a known value less than 64
  #pragma omp target teams num_teams(teams) thread_limit(SX*64)
  {
    float local_result[SX * SY];
    float local_targets[SX * 3];
    float local_sources[SY * 3];
    float local_weights[SY];

    #pragma omp parallel
    {
      int tx = omp_get_thread_num() % SX;
      int ty = omp_get_thread_num() / SX;
      int px = omp_get_team_num() * SX + tx; // range [0:ntargets)
      int py = ty; // range [0:nsources)

      if (px < ntargets && py < SY) {
        if (ty == 0) {
          for (int i = 0; i < 3; i++)
            local_targets[tx * 3 + i] = targets[px * 3 + i];
        }

        if (tx == 0) {
          for (int i = 0; i < 3; i++)
            local_sources[ty * 3 + i] = sources[py * 3 + i];
          local_weights[ty] = weights[ty];
        }
      }
      #pragma omp barrier

      if (px < ntargets && py < SY) {
        float squared_diff = 0.f;
        
        for (int i = 0; i < 3; i++) {
          squared_diff += (local_targets[tx * 3 + i] - local_sources[ty * 3 + i]) *
                          (local_targets[tx * 3 + i] - local_sources[ty * 3 + i]);
        }
        float diff = sqrtf(squared_diff);

        local_result[tx * SY + ty] = 
          (1.f + sqrtf(5.f) * diff / l + 5.f * squared_diff / (3.f * l * l)) *  
          expf(-sqrtf(5.f) * diff / l) * local_weights[ty];

      }
      #pragma omp barrier

      if (px < ntargets && py < SY) {
        if (ty == 0) {
          float res = 0.f;
          for (int i = 0; i < SY; i++)
            res += local_result[tx * SY + i];
          result[px] = res;
        }
      }
    }
  }
  #pragma omp target update from(result[0:ntargets])
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of points> <repeat>\n", argv[0]);
    return 1;
  }
  const int npoints = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  const int source_size = nsources * 3;  // (x,y,z) coordinates in a 3D grid
  const int source_size_byte = source_size * sizeof(float);

  const int weight_size = nsources;
  const int weight_size_byte = weight_size * sizeof(float);

  const int ntargets = npoints * npoints * npoints;
  const int target_size = ntargets * 3;
  const int target_size_byte = target_size * sizeof(float);

  const int result_size = ntargets;
  const int result_size_byte = ntargets * sizeof(float);

  float *sources = (float*) malloc (source_size_byte);
  float *targets = (float*) malloc (target_size_byte);
  float *weights = (float*) malloc (weight_size_byte);
  float *result = (float*) malloc (result_size_byte);
  float *result_ref = (float*) malloc (result_size_byte);

  srand(123);
  for (int i = 0; i < source_size; i++)
    sources[i] = rand() / (float)RAND_MAX;

  for (int i = 0; i < weight_size; i++)
    weights[i] = rand() / (float)RAND_MAX;

  for (int i = 0; i < target_size; i++) 
    targets[i] = rand() / (float)RAND_MAX;

  #pragma omp target data map(to: sources[0:source_size],\
                                  weights[0:weight_size],\
                                  targets[0:target_size]) \
                          map(alloc: result[0:result_size])
  {
    float l = 0.1f; // length scale lower bound

    // quickly verify the results using a small problem size
    const int ntargets_small = 16*16*16;
    printf("------------------------------------------------------------\n");
    printf("Verifying the kernel results with the problem size (16 cube)\n");
    printf("------------------------------------------------------------\n");

    while (l <= 1e5f) {
      matern_kernel_reference(nsources, ntargets_small, l, sources, targets, weights, result_ref);

      matern_kernel2(ntargets_small, l, sources, targets, weights, result);

      bool ok = true;
      for (int i = 0; i < ntargets_small; i++) {
        if (fabsf(result[i] - result_ref[i]) > 1e-3f) {
          printf("@%d actual=%f expected=%f\n", i, result[i] , result_ref[i]);
          ok = false;
          break;
        }
      }
      printf("Length scale = %.1e check = %s\n", l, ok ? "PASS" : "FAIL");
      l = l * 10.f;
    }

    printf("--------------------------------------------------------------------\n");
    printf("Timing the kernel execution with the problem size (%d cube)\n", npoints);
    printf("--------------------------------------------------------------------\n");

    l = 0.1f;
    while (l <= 1e5f) {
      printf("Warmup..\n");
      for (int i = 0; i < repeat; i++) {
        matern_kernel2(ntargets, l, sources, targets, weights, result);
      }

      auto start = std::chrono::steady_clock::now();

      for (int i = 0; i < repeat; i++) {
        matern_kernel2(ntargets, l, sources, targets, weights, result);
      }

      auto end = std::chrono::steady_clock::now();
      auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Length scale = %.1e ", l);
      printf("Average kernel execution time: %f (us)\n", (time * 1e-3f) / repeat);

      l = l * 10.f;
    }
  }

  free(sources);
  free(weights);
  free(targets);
  free(result);
  free(result_ref);
  if (!ok) return 1;
  return 0;
}
