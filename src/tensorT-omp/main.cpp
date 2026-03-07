#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <omp.h>

#define TILE_SIZE 5900
#define NTHREADS 256

// 1,2,3,4,5,6 -> 2,3,4,6,1,5
static const int d1 = 41, d2 = 13, d3 = 11, d4 = 9, d5 = 76, d6 = 50;
static const int data_size = d1 * d2 * d3 * d4 * d5 * d6;
static int repeat = 1;

static const int shape_output[] = {d2, d3, d1};
static const int shape_input[] = {d4, d5, d6};
static const float shape_output_r[] = {1.0 / d2, 1.0 / d3, 1.0 / d1};
static const float shape_input_r[] = {1.0 / d4, 1.0 / d5, 1.0 / d6};
static const int stride_output_local[] = {d1, d1 * d2, 1};
static const int stride_output_global[] = {1, d2, d2 * d3 * d4 * d6};
static const int stride_input[] = {d2 * d3, d2 * d3 * d4 * d6 * d1, d2 * d3 * d4};

void verify(double *input, double *output) {
  int input_offset  = 2 + d1 * (2 + d2 * (2 + d3 * (2 + d4 * (0 + 2 * d5))));
  int output_offset = 2 + d2 * (2 + d3 * (2 + d4 * (2 + d6 * (2 + 0 * d1))));
  bool error = false;
  for (size_t i = 0; i < d5; i++) {
    if (input[input_offset + i * d1 * d2 * d3 * d4] != 
        output[output_offset + i * d2 * d3 * d4 * d6 * d1]) {
      printf("FAIL\n");
      error = true;
      break;
    }
  }
  if (!error) printf("PASS\n");
}

int main(int argc, char **argv) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  repeat = atoi(argv[1]);

  double *input = new double[data_size]();
  double *output = new double[data_size]();

  for (size_t i = 0; i < data_size; i++) {
    input[i] = i;
  }

  const int nblocks = d4 * d5 * d6;
  const int tile_size = d1 * d2 * d3;
  const int dim_output = 3;
  const int dim_input = 3;

#pragma omp target data map(to: input[0:data_size],  \
                                shape_input[0:dim_input], \
                                shape_input_r[0:dim_input], \
                                shape_output[0:dim_output], \
                                shape_output_r[0:dim_output], \
                                stride_input[0:dim_input], \
                                stride_output_local[0:dim_output], \
                                stride_output_global[0:dim_output]) \
                        map(from: output[0:data_size])
  {
    auto start = std::chrono::steady_clock::now();

    for (size_t i = 0; i < repeat; ++i) {

      #pragma omp target teams num_teams(nblocks) thread_limit(NTHREADS)
      {
        double tile[TILE_SIZE];
        #pragma omp parallel 
        {
          for (int block_idx = omp_get_team_num(); block_idx < nblocks; block_idx += omp_get_num_teams()) {
            int it = block_idx, im = 0, offset1 = 0;
            for (int i = 0; i < dim_input; i++) {
              im = it * shape_input_r[i];
              offset1 += stride_input[i] * (it - im * shape_input[i]);
              it = im;
            }

            for (int i = omp_get_thread_num(); i < tile_size; i += omp_get_num_threads()) {
              tile[i] = input[i + block_idx * tile_size];
            }

            #pragma omp barrier

            for (int i = omp_get_thread_num(); i < tile_size; i += omp_get_num_threads()) {
              it = i;
              int offset2 = 0, local_offset = 0;
              for (int j = 0; j < dim_output; j++) {
                im = it * shape_output_r[j];
                int tmp = it - im * shape_output[j];
                offset2 += stride_output_global[j] * tmp;
                local_offset += stride_output_local[j] * tmp;
                it = im;
              }
              output[offset1 + offset2] = tile[local_offset];
            }
            #pragma omp barrier
          }
        }
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);
  }

  verify(input, output);
  delete [] input;
  delete [] output;
  if (error) return 1;
  return 0;
}
