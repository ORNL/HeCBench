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
  if (repeat < 2) {
    printf("repeat must be at least 2\n");
    return 1;
  }

  const size_t num_Apoints_bytes = sizeof(float2) * num_Apoints;
  const size_t num_Bpoints_bytes = sizeof(float2) * num_Bpoints;

  float2 *h_A[2], *h_B[2];
  for (int b = 0; b < 2; b++) {
    h_A[b] = (float2*) malloc(num_Apoints_bytes);
    h_B[b] = (float2*) malloc(num_Bpoints_bytes);
  }

  srand(123);
  for (int b = 0; b < 2; b++) {
    for (int i = 0; i < num_Apoints; i++) {
      h_A[b][i].x = (float)rand() / (float)RAND_MAX;
      h_A[b][i].y = (float)rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < num_Bpoints; i++) {
      h_B[b][i].x = (float)rand() / (float)RAND_MAX;
      h_B[b][i].y = (float)rand() / (float)RAND_MAX;
      // Slot 0 is shifted so its distance stays above slot 1 for any point
      // count. Iteration 1 then keeps slot 0 whenever -1 was uploaded once.
      if (b == 0) {
        h_B[b][i].x += 10.f;
        h_B[b][i].y += 10.f;
      }
    }
  }

  // The max reduction keeps the value already at this address. A single
  // upload of -1 therefore leaves the larger slot in place after iteration 0.
  float h_distance[2] = {-1.f, -1.f};
  float got[2];

#pragma omp target data map (to: h_A[0][0:num_Apoints], h_A[1][0:num_Apoints], \
                                 h_B[0][0:num_Bpoints], h_B[1][0:num_Bpoints]) \
                        map (alloc: h_distance[0:2])
  {
    double time = 0.0;

    for (int i = 0; i < repeat; i++) {
      const int slot = i & 1;

      h_distance[0] = h_distance[1] = -1.f;
      #pragma omp target update to (h_distance[0:2])

      auto start = std::chrono::steady_clock::now();

      computeDistance(h_A[slot], h_B[slot], h_distance, num_Apoints, num_Bpoints);
      computeDistance(h_B[slot], h_A[slot], h_distance + 1, num_Bpoints, num_Apoints);

      auto end = std::chrono::steady_clock::now();
      time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

      // The two slots are fully covered by the first two iterations.
      if (i < 2) {
        #pragma omp target update from (h_distance[0:2])
        got[i] = std::max(h_distance[0], h_distance[1]);
      }
    }

    printf("Average execution time of kernels: %f (ms)\n", (time * 1e-6f) / repeat);
  }

  printf("Verifying the result may take a while..\n");
  bool error = false;
  for (int s = 0; s < 2; s++) {
    float r_distance = hausdorff_distance(h_A[s], h_B[s], num_Apoints, num_Bpoints);
    if (fabsf(got[s] - r_distance) > 1e-3f * r_distance)
      error = true;
  }
  printf("%s\n", error ? "FAIL" : "PASS");

  for (int b = 0; b < 2; b++) {
    free(h_A[b]);
    free(h_B[b]);
  }
  return 0;
}
