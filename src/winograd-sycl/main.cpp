#include <chrono>
#include <sycl/sycl.hpp>
#include "utils.h"
#include <cstdlib>

int main(int argc, char* argv[]) {

  DATA_TYPE *A = (DATA_TYPE*)malloc(MAP_SIZE * MAP_SIZE * sizeof(DATA_TYPE));
  DATA_TYPE *B = (DATA_TYPE*)malloc((MAP_SIZE - 2) * (MAP_SIZE - 2) * sizeof(DATA_TYPE));
  DATA_TYPE *B_outputFromGpu = (DATA_TYPE*)malloc((MAP_SIZE - 2) * (MAP_SIZE - 2) * sizeof(DATA_TYPE));
  DATA_TYPE *C = (DATA_TYPE*)malloc(4 * 4 * sizeof(DATA_TYPE));

  for (int i = 0; i < MAP_SIZE; ++i)
    for (int j = 0; j < MAP_SIZE; ++j)
      A[i * MAP_SIZE + j] = rand() / (float)RAND_MAX;

  // transformed filter
  WinogradConv2D_2x2_filter_transformation(C);

#ifdef USE_GPU
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
#else
  sycl::queue q(sycl::cpu_selector_v, sycl::property::queue::in_order());
#endif

  double start = rtclock();

  DATA_TYPE *d_A = sycl::malloc_device<DATA_TYPE>(MAP_SIZE * MAP_SIZE, q);
  q.memcpy(d_A, A, MAP_SIZE * MAP_SIZE * sizeof(DATA_TYPE));

  DATA_TYPE *d_B = sycl::malloc_device<DATA_TYPE>((MAP_SIZE-2) * (MAP_SIZE-2), q);

  DATA_TYPE *d_C = sycl::malloc_device<DATA_TYPE>(16, q);
  q.memcpy(d_C, C, 16 * sizeof(DATA_TYPE));

  const int tile_n = (MAP_SIZE - 2 + 1) / 2;

  // initial problem size
  size_t globalWorkSize[2] = {
    (size_t)ceil(((float)tile_n) / ((float)DIM_LOCAL_WORK_GROUP_X)) * DIM_LOCAL_WORK_GROUP_X,
    (size_t)ceil(((float)tile_n) / ((float)DIM_LOCAL_WORK_GROUP_Y)) * DIM_LOCAL_WORK_GROUP_Y };

  size_t localWorkSize[2] = {DIM_LOCAL_WORK_GROUP_X, DIM_LOCAL_WORK_GROUP_Y};

  // adjust problem size for co-run
  size_t cpu_global_size[2];
  size_t gpu_global_size[2];

  bool pass = true;

  // sweep over cpu_offset
  double co_time = 0.0;

  for (int cpu_offset = 0; cpu_offset <= 100; cpu_offset++) {

    cpu_global_size[0] = cpu_offset * (size_t)ceil(((float)tile_n) / ((float)DIM_LOCAL_WORK_GROUP_X))
      / 100 * DIM_LOCAL_WORK_GROUP_X;
    cpu_global_size[1] = globalWorkSize[1];
    gpu_global_size[0] = globalWorkSize[0] - cpu_global_size[0];
    gpu_global_size[1] = globalWorkSize[1];

    const int offset_j = cpu_global_size[0];
    const int offset_i = 0;

    sycl::range<2> gpu_gws(gpu_global_size[1], gpu_global_size[0]);
    sycl::range<2> cpu_gws(cpu_global_size[1], cpu_global_size[0]);
    sycl::range<2> lws(localWorkSize[1], localWorkSize[0]);

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
      q.submit([&] (sycl::handler &cgh) {
        cgh.parallel_for<class winograd_conv2d>(
          sycl::nd_range<2>(gpu_gws, lws), [=] (sycl::nd_item<2> item) {

          int tile_j = item.get_global_id(0) + offset_j;
          int tile_i = item.get_global_id(1) + offset_i;

          // d_A transformation

          DATA_TYPE d_A_tile[4][4], tmp_tile[4][4], transformed_tile[4][4];
          for (int i = 0; i < 4; i ++) {
            for (int j = 0; j < 4; j ++) {
              int x = 2 * tile_i + i;
              int y = 2 * tile_j + j;
              if (x >= MAP_SIZE || y >= MAP_SIZE) {
                d_A_tile[i][j] = 0;
                continue;
              }
              d_A_tile[i][j] = d_A[x * MAP_SIZE + y];
            }
          }

          // Bt * d
          for (int j = 0; j < 4; j ++) {
            tmp_tile[0][j] = d_A_tile[0][j] - d_A_tile[2][j];
            tmp_tile[1][j] = d_A_tile[1][j] + d_A_tile[2][j];
            tmp_tile[2][j] = -d_A_tile[1][j] + d_A_tile[2][j];
            tmp_tile[3][j] = d_A_tile[1][j] - d_A_tile[3][j];
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
              multiplied_tile[i][j] = transformed_tile[i][j] * d_C[i * 4 + j];
            }
          }

          // d_B transformation

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
              int x = 2 * tile_i + i;
              int y = 2 * tile_j + j;
              if (x >= MAP_SIZE - 2 || y >= MAP_SIZE - 2) {
                continue;
              }
              d_B[x * (MAP_SIZE - 2) + y] = final_tile[i][j];
            }
          }
        });
      });

    }

    if (cpu_run) {

      // printf("CPU size: %d\n", cpu_global_size[0]);
      WinogradConv2D_2x2_omp(A, B, C, cpu_global_size);

      q.memcpy(d_B, B, gpu_run ? cpu_global_size[0]*2*(MAP_SIZE-2)*sizeof(DATA_TYPE) : 
               (MAP_SIZE-2)*(MAP_SIZE-2)*sizeof(DATA_TYPE));
    }

    q.memcpy(B_outputFromGpu, d_B, (MAP_SIZE-2) * (MAP_SIZE-2) * sizeof(DATA_TYPE)).wait();

    co_time += rtclock() - co_start;

#ifdef VERBOSE
    if (cpu_run) printf("run on host\n");
    if (gpu_run) printf("run on device\n");
    printf("CPU workload size : %d\n", cpu_offset);
#endif

    WinogradConv2D_2x2(A, B, C);
    pass &= compareResults(B, B_outputFromGpu);

  } // sweep

  printf("%s\n", pass ? "PASS" : "FAIL");

  sycl::free(d_A, q);
  sycl::free(d_B, q);
  sycl::free(d_C, q);
  free(A);
  free(B);
  free(B_outputFromGpu);
  free(C);

  double end = rtclock();
  printf("Co-execution time: %lf s\n", co_time);
  printf("Total time: %lf s\n", end - start);
  printf("Ratio of co-execution time to total time: %.2lf%%\n",
         100.0 * co_time / (end - start));

  if (!pass) return 1;
  return 0;
}
