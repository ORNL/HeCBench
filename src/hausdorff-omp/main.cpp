#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <omp.h>
#include "reference.h"

inline float hd (const float2 ap, const float2 bp)
{
  return (ap.x - bp.x) * (ap.x - bp.x)
       + (ap.y - bp.y) * (ap.y - bp.y);
}

void computeDistance(const float2* __restrict Apoints,
                     const float2* __restrict Bpoints,
                           float*  __restrict distance,
                     const int numA, const int numB)
{
  #pragma omp target teams distribute parallel for \
   reduction(max:distance[0]) thread_limit(256) 
  for (int i = 0; i < numA; i++) {
    float d = FLT_MAX;
    float2 p = Apoints[i];
    for (int j = 0; j < numB; j++)
    {
      float t = hd(p, Bpoints[j]);
      d = std::min(t, d);
    }
    distance[0] = std::max(distance[0], d);
  }
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <number of points in space A>", argv[0]);
    printf(" <number of points in space B> <repeat>\n");
    return 1;
  }
  const int num_Apoints = atoi(argv[1]);
  const int num_Bpoints = atoi(argv[2]);
  const int repeat = atoi(argv[3]);

  const size_t num_Apoints_bytes = sizeof(float2) * num_Apoints;
  const size_t num_Bpoints_bytes = sizeof(float2) * num_Bpoints;

  float2 *h_Apoints = (float2*) malloc (num_Apoints_bytes);
  float2 *h_Bpoints = (float2*) malloc (num_Bpoints_bytes);
  
  srand(123);
  for (int i = 0; i < num_Apoints; i++) {
    h_Apoints[i].x = (float)rand() / (float)RAND_MAX;
    h_Apoints[i].y = (float)rand() / (float)RAND_MAX;
  }
  
  for (int i = 0; i < num_Bpoints; i++) {
    h_Bpoints[i].x = (float)rand() / (float)RAND_MAX;
    h_Bpoints[i].y = (float)rand() / (float)RAND_MAX;
  }

  float h_distance[2] = {-1.f, -1.f};

#pragma omp target data map (to: h_Apoints[0:num_Apoints], \
                                 h_Bpoints[0:num_Bpoints]) \
                        map (from: h_distance[0:2]) 
  {
    double time = 0.0;

    for (int i = 0; i < repeat; i++) {

      #pragma omp target update to (h_distance[0:2])

      auto start = std::chrono::steady_clock::now();

      computeDistance(h_Apoints, h_Bpoints, h_distance,
                      num_Apoints, num_Bpoints);

      computeDistance(h_Bpoints, h_Apoints, h_distance+1,
                      num_Bpoints, num_Apoints);

      auto end = std::chrono::steady_clock::now();
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }

    printf("Average execution time of kernels: %f (ms)\n", (time * 1e-6f) / repeat);
  }

  printf("Verifying the result may take a while..\n");
  float r_distance = hausdorff_distance(h_Apoints, h_Bpoints, num_Apoints, num_Bpoints);
  float t_distance = std::max(h_distance[0], h_distance[1]);

  bool error = (fabsf(t_distance - r_distance)) > 1e-3f;
  printf("%s\n", error ? "FAIL" : "PASS");
  if (error) exit(1);
  
  free(h_Apoints);
  free(h_Bpoints);
  return 0;
}
