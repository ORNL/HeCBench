#include <chrono>
#include <omp.h>
#include "utils.h"
#include <cstdlib>

int main(int argc, char* argv[]) {

  double start = rtclock();

  DATA_TYPE *A = (DATA_TYPE*)malloc(MAP_SIZE * MAP_SIZE * sizeof(DATA_TYPE));
  DATA_TYPE *B_host = (DATA_TYPE*)malloc((MAP_SIZE - 2) * (MAP_SIZE - 2) * sizeof(DATA_TYPE));
  DATA_TYPE *B = (DATA_TYPE*)malloc((MAP_SIZE - 2) * (MAP_SIZE - 2) * sizeof(DATA_TYPE));
  DATA_TYPE *C = (DATA_TYPE*)malloc(4 * 4 * sizeof(DATA_TYPE));

  for (int i = 0; i < MAP_SIZE; ++i)
    for (int j = 0; j < MAP_SIZE; ++j)
      A[i * MAP_SIZE + j] = rand() / (float)RAND_MAX;

  // transformed filter
  WinogradConv2D_2x2_filter_transformation(C);

  const int tile_n = (MAP_SIZE - 2 + 1) / 2;

  // initial problem size
  size_t globalWorkSize[2] = {
    (size_t)ceil(((float)tile_n) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X,
    (size_t)ceil(((float)tile_n) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y };

  size_t localWorkSize[2] = {DIM_LOCAL_WORK_GROUP_X, DIM_LOCAL_WORK_GROUP_Y};

  // adjust problem size for co-run
  size_t cpu_global_size[2];
  size_t gpu_global_size[2];
  size_t global_offset[2];

  bool pass = true;

  double co_time = 0.0;

#pragma omp target data map (to: A[0:MAP_SIZE * MAP_SIZE],C[0:16]), \
                        map (alloc: B[0:(MAP_SIZE-2) * (MAP_SIZE-2)])

{
  // sweep over cpu workload size
  for (int cpu_offset = 0; cpu_offset <= 100; cpu_offset++) {

    cpu_global_size[0] = cpu_offset * (size_t)ceil(((float)tile_n) / ((float)DIM_LOCAL_WORK_GROUP_X)) 
      / 100 * DIM_LOCAL_WORK_GROUP_X;
    cpu_global_size[1] = globalWorkSize[1];

    gpu_global_size[0] = globalWorkSize[0] - cpu_global_size[0];
    gpu_global_size[1] = globalWorkSize[1];

    global_offset[0] = cpu_global_size[0];
    global_offset[1] = 0;

    const int tile_i_size = gpu_global_size[0];
    const int tile_j_size = gpu_global_size[1];
    const int offset_i = global_offset[0];
    const int offset_j = global_offset[1];
    const int thread_size = localWorkSize[1] * localWorkSize[0];

    bool cpu_run = false, gpu_run = false;
    if (cpu_global_size[0] > 0) {
      cpu_run = true;
    }
    if (gpu_global_size[0] > 0) {
      gpu_run = true;
    }

    // co-execution of host and device
    double co_start = rtclock();

    if (gpu_run) {
      #pragma omp target teams distribute parallel for collapse(2) thread_limit(thread_size)
      for (int tile_j = 0; tile_j < tile_j_size; tile_j++) {
        for (int tile_i = 0; tile_i < tile_i_size; tile_i++) {
          // input transformation

          DATA_TYPE input_tile[4][4], tmp_tile[4][4], transformed_tile[4][4];
          for (int i = 0; i < 4; i ++) {
            for (int j = 0; j < 4; j ++) { 
              int x = 2 * (tile_i + offset_i) + i;
              int y = 2 * (tile_j + offset_j) + j;
              if (x >= MAP_SIZE || y >= MAP_SIZE) {
                input_tile[i][j] = 0;
                continue;
              }
              input_tile[i][j] = A[x * MAP_SIZE + y];
            }
          } 

          // Bt * d
          for (int j = 0; j < 4; j ++) {
            tmp_tile[0][j] = input_tile[0][j] - input_tile[2][j];
            tmp_tile[1][j] = input_tile[1][j] + input_tile[2][j];
            tmp_tile[2][j] = -input_tile[1][j] + input_tile[2][j];
            tmp_tile[3][j] = input_tile[1][j] - input_tile[3][j];
          }
          // d * B
          for (int i = 0; i < 4; i ++) {
            transformed_tile[i][0] = tmp_tile[i][0] - tmp_tile[i][2];
            transformed_tile[i][1] = tmp_tile[i][1] + tmp_tile[i][2];
            transformed_tile[i][2] = -tmp_tile[i][1] + tmp_tile[i][2];
            transformed_tile[i][3] = tmp_tile[i][1] - tmp_tile[i][3];
          }

          // element-wise multiplication

          DATA_TYPE multiplied_tile[4][4];
          for (int i = 0; i < 4; i ++) {
            for (int j = 0; j < 4; j ++) {
              multiplied_tile[i][j] = transformed_tile[i][j] * C[i * 4 + j];
            }
          }

          // output transformation

          DATA_TYPE tmp_tile_1[2][4], final_tile[2][2];

          // At * I
          for (int j = 0; j < 4; j ++) {
            tmp_tile_1[0][j] = multiplied_tile[0][j] + multiplied_tile[1][j] + multiplied_tile[2][j];
            tmp_tile_1[1][j] = multiplied_tile[1][j] - multiplied_tile[2][j] - multiplied_tile[3][j];
          }
          // I * A
          for (int i = 0; i < 2; i ++) {
            final_tile[i][0] = tmp_tile_1[i][0] + tmp_tile_1[i][1] + tmp_tile_1[i][2];
            final_tile[i][1] = tmp_tile_1[i][1] - tmp_tile_1[i][2] - tmp_tile_1[i][3];
          }

          for (int i = 0; i < 2; i ++) {
            for (int j = 0; j < 2; j ++) {
              int x = 2 * (tile_i + offset_i) + i;
              int y = 2 * (tile_j + offset_j) + j;
              if (x >= MAP_SIZE - 2 || y >= MAP_SIZE - 2) {
                continue;
              }
              B[x * (MAP_SIZE - 2) + y] = final_tile[i][j];
            }
          }
        }
      }
    }

    if (cpu_run) {
      WinogradConv2D_2x2_omp(A, B, C, cpu_global_size);

      if (gpu_run) {
        #pragma omp target update to (B[0:offset_i*2*(MAP_SIZE-2)])
      }
      else {
        #pragma omp target update to (B[0:(MAP_SIZE-2)*(MAP_SIZE-2)])
      }
    }

    #pragma omp target update from (B[0:(MAP_SIZE-2)*(MAP_SIZE-2)])

    co_time += rtclock() - co_start;

#ifdef VERBOSE
    if (cpu_run) printf("run on host\n");
    if (gpu_run) printf("run on device\n");
    printf("CPU workload size : %d\n", cpu_offset);
#endif

    WinogradConv2D_2x2(A, B_host, C);
    pass &= compareResults(B_host, B);

  } // sweep
}  // #pragma

  printf("%s\n", pass ? "PASS" : "FAIL");
  if (!pass) exit(1);

  free(A);
  free(B);
  free(B_host);
  free(C);

  double end = rtclock();
  printf("Co-execution time: %lf s\n", co_time);
  printf("Total time: %lf s\n", end - start);
  printf("Ratio of co-execution time to total time: %.2lf%%\n",
         100.0 * co_time / (end - start));

  return 0;
}
