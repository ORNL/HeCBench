#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <omp.h>

#define NUM_OF_BLOCKS (1024 * 1024)
#define NUM_OF_THREADS 128

#define EXPECTED 65504.f

using __half = _Float16;

void generateInput(__half* x, __half* y, size_t size)
{
  float val = (float)std::sqrt(32752.0 / (double)size);
  for (size_t i = 0; i < size; ++i) {
    x[i] = (__half)val;
    y[i] = (__half)val;
  }
}

// FP16 shared-memory tree reduction
void kernel1(const __half* __restrict__ ax,
             const __half* __restrict__ ay,
             const __half* __restrict__ bx,
             const __half* __restrict__ by,
             float*       __restrict__ result,
             size_t size, int grid)
{
  #pragma omp target teams num_teams(grid)
  {
    __half sh_x[NUM_OF_THREADS];
    __half sh_y[NUM_OF_THREADS];

    #pragma omp parallel num_threads(NUM_OF_THREADS)
    {
      int lid    = omp_get_thread_num();
      size_t stride = (size_t)grid * NUM_OF_THREADS;
      size_t base   = (size_t)omp_get_team_num() * NUM_OF_THREADS + lid;

      __half val_x = 0, val_y = 0;
      for (size_t i = base; i < size; i += stride) {
        val_x += ax[i] * bx[i];
        val_y += ay[i] * by[i];
      }
      sh_x[lid] = val_x;
      sh_y[lid] = val_y;

      #pragma omp barrier

      for (int i = NUM_OF_THREADS / 2; i >= 1; i /= 2) {
        if (lid < i) {
          sh_x[lid] += sh_x[lid + i];
          sh_y[lid] += sh_y[lid + i];
        }
        #pragma omp barrier
      }

      if (lid == 0) {
        float f_result = (float)sh_x[0] + (float)sh_y[0];
        #pragma omp atomic
        result[0] += f_result;
      }
    }
  }
}

// FP32 shared-memory tree reduction
void kernel2(const __half* __restrict__ ax,
             const __half* __restrict__ ay,
             const __half* __restrict__ bx,
             const __half* __restrict__ by,
             float*       __restrict__ result,
             size_t size, int grid)
{
  #pragma omp target teams num_teams(grid)
  {
    float sh_x[NUM_OF_THREADS];
    float sh_y[NUM_OF_THREADS];

    #pragma omp parallel num_threads(NUM_OF_THREADS)
    {
      int lid    = omp_get_thread_num();
      size_t stride = (size_t)grid * NUM_OF_THREADS;
      size_t base   = (size_t)omp_get_team_num() * NUM_OF_THREADS + lid;

      float val_x = 0.f, val_y = 0.f;
      for (size_t i = base; i < size; i += stride) {
        val_x += float(ax[i]) * float(bx[i]);
        val_y += float(ay[i]) * float(by[i]);
      }
      sh_x[lid] = val_x;
      sh_y[lid] = val_y;

      #pragma omp barrier

      for (int i = NUM_OF_THREADS / 2; i >= 1; i /= 2) {
        if (lid < i) {
          sh_x[lid] += sh_x[lid + i];
          sh_y[lid] += sh_y[lid + i];
        }
        #pragma omp barrier
      }

      if (lid == 0) {
        #pragma omp atomic
        result[0] += sh_x[0] + sh_y[0];
      }
    }
  }
}

__half dotProduct(const __half* ax, const __half* ay,
                  const __half* bx, const __half* by,
                  size_t size)
{
  float dot = 0.f;
  #pragma omp target teams distribute parallel for reduction(+:dot)
  for (size_t i = 0; i < size; ++i)
    dot += float(ax[i] * bx[i]) + float(ay[i] * by[i]);
  return dot;
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = std::atoi(argv[1]);

  const size_t size = (size_t)NUM_OF_BLOCKS * NUM_OF_THREADS;
  const size_t size_bytes = size * sizeof(__half);

  __half* ax = (__half*)malloc(size_bytes);
  __half* ay = (__half*)malloc(size_bytes);
  __half* bx = (__half*)malloc(size_bytes);
  __half* by = (__half*)malloc(size_bytes);

  generateInput(ax, ay, size);
  generateInput(bx, by, size);

  printf("\nNumber of elements in the vectors is %zu\n", size * 2);

  float result[1];

  #pragma omp target data map(to: ax[0:size], ay[0:size],\
                                  bx[0:size], by[0:size]) \
                          map(alloc: result[0:1])
  {
    // Evaluate the impact of grid sizes on performance (mirrors original loop)
    for (int grid = NUM_OF_BLOCKS; grid >= NUM_OF_BLOCKS / 16; grid /= 2) {

      printf("\nGPU grid size (num_teams) is %d\n", grid);

      // Kernel 1 warmup
      for (int i = 0; i < 1000; ++i) { kernel1(ax,ay,bx,by,result,size,grid); }

      auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < repeat; ++i) {
        #pragma omp target
        result[0] = 0.f;
        kernel1(ax,ay,bx,by,result,size,grid);
      }
      auto end  = std::chrono::steady_clock::now();
      auto time   = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average kernel1 execution time %f (us)\n", (time * 1e-3f) / repeat);
      #pragma omp target update from (result[0:1])
      printf("Error rate: %e\n", fabsf(result[0] - EXPECTED) / EXPECTED);

      // Kernel 2 warmup
      for (int i = 0; i < 1000; ++i) { kernel2(ax,ay,bx,by,result,size,grid); }

      start = std::chrono::steady_clock::now();
      for (int i = 0; i < repeat; ++i) {
        #pragma omp target
        result[0] = 0.f;
        kernel2(ax,ay,bx,by,result,size,grid);
      }
      end = std::chrono::steady_clock::now();
      time  = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average kernel2 execution time %f (us)\n", (time * 1e-3f) / repeat);
      #pragma omp target update from (result[0:1])
      printf("Error rate: %e\n", fabsf(result[0] - EXPECTED) / EXPECTED);
    }

    printf("\n");
    for (int i = 0; i < 1000; ++i) dotProduct(ax, ay, bx, by, size);

    auto start = std::chrono::steady_clock::now();
    float dotResult = 0.f;
    for (int i = 0; i < repeat; ++i)
      dotResult = dotProduct(ax, ay, bx, by, size);
    auto end = std::chrono::steady_clock::now();
    auto time  = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel (dot reduction) execution time %f (us)\n", (time * 1e-3f) / repeat);
    printf("Error rate: %e\n", fabsf(dotResult - EXPECTED) / EXPECTED);
  }

  free(ax);
  free(ay);
  free(bx);
  free(by);
  return EXIT_SUCCESS;
}
