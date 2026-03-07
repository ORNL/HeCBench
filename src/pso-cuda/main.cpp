#include "kernel.h"
#include <cstdlib>

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of particles> <repeat>\n", argv[0]);
    return 1;
  }
  const int p = atoi(argv[1]);
  const int r = atoi(argv[2]);

  printf("Number of particles is %d\n", p);
  printf("Number of dimensions is %d\n", DIM);

  size_t size = p * DIM;
  size_t size_byte = sizeof(float) * size;

  float *positions = (float*) malloc (size_byte);
  float *velocities = (float*) malloc (size_byte);
  float *pBests = (float*) malloc (size_byte);
  float *pBests_ref = (float*) malloc (size_byte);
  float *gBest = (float*) malloc (sizeof(float) * DIM);

  srand(123);

  for(size_t i=0;i<size;i++)
  {
    positions[i]=getRandom(START_RANGE_MIN,START_RANGE_MAX);
    pBests[i]=pBests_ref[i]=positions[i];
    velocities[i]=0;
  }

  for(int i=0;i<DIM;i++) gBest[i]=pBests[i];

  printf("\nExecute PSO on a device\n");
  gpu_pso(p,r,positions,velocities,pBests,gBest);

  // The result of floating-point atomic adds
  printf("Result=%f\n",host_fitness_function(gBest));

  printf("\nExecute PSO on a host. This may take a while for large problem size..\n");

  for(int i=0;i<DIM;i++) gBest[i]=pBests_ref[i];

  pso(p,r,positions,velocities,pBests_ref,gBest);

  // The result of sequential floating-point adds
  printf("Result=%f\n",host_fitness_function(gBest));

  bool ok = true;
  for(int i=0;i<DIM;i++) {
    if (fabsf(pBests_ref[i] - pBests[i]) > 1e-3f) {
      printf("@%d %f %f\n", i, pBests_ref[i], pBests[i]);
      ok = false;
      break;
    }
  }

  printf("%s\n", ok ? "PASS" : "FAIL");
  if (!ok) exit(1);

  free(positions);
  free(velocities);
  free(pBests);
  free(pBests_ref);
  free(gBest);
  return 0;
}
