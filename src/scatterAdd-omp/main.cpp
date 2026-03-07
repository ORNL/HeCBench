#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <omp.h>
#include "kernels.h"

/*
Reference
vip-token-centric-compression/src/t5/models/small_embedding/kernel.py

batch size = 4
vector dim = 5
>>> indexes = torch.randint(0, 3, size = (4, ))
tensor([1, 1, 2, 0])
>>> indexes = indexes[:, None].repeat(1, 5)
>>> indexes
tensor([[1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
        [0, 0, 0, 0, 0]])
outputs.scatter_add_(0, indexes, source)
*/
void scatter_add_reference (int batch_size, int vector_dim,
                            float *out, int *idx, float *src)
{
  for (int d = 0; d < vector_dim; d++) {
    for (int i = 0; i < batch_size; i++) {
      int index = idx[i];
      out[index * vector_dim + d] += src[i * vector_dim + d];
    }
  }
}

void index_accumulate (int batch_size, int output_size, int vector_dim, int repeat)
{
  size_t source_size_bytes = batch_size * vector_dim * sizeof(float);
  size_t output_size_bytes = output_size * vector_dim * sizeof(float);
  size_t index_size_bytes = batch_size * sizeof(int);

  int* index = (int*) malloc (index_size_bytes);
  float* source = (float*) malloc (source_size_bytes);
  float* output = (float*) malloc (output_size_bytes);
  float* output_ref = (float*) malloc (output_size_bytes);

  srand(2);
  for (int i = 0; i < batch_size; i++) {
    index[i] = rand() % output_size;
  }

  for (int i = 0; i < batch_size * vector_dim; i++) {
    source[i] = -1.f; // or random values
  }

  memset(output_ref, 0, output_size_bytes);
  scatter_add_reference (batch_size, vector_dim, output_ref, index, source);

  #pragma omp target data map(to: source[0:batch_size * vector_dim],\
                                  index[0:batch_size]) \
                          map(alloc: output[0:output_size * vector_dim])
  {
    // verify and warmup
    for (int i = 0; i < 10; i++) {
      memset(output, 0, output_size_bytes);
      #pragma omp target update to (output[0:output_size * vector_dim])
      scatterAdd2_kernel(
        index,
        source,
        output,
        batch_size,
        output_size,
        vector_dim
      );
    }

    #pragma omp target update from (output[0:output_size * vector_dim])
    bool ok = true;
    for (int i = 0; i < output_size * vector_dim; i++) {
      if (fabsf(output[i] - output_ref[i]) > 1e-3f) {
        printf("output %d: %f %f\n", i, output[i], output_ref[i]);
        ok = false;
        break;
      }
    }
    printf("%s\n", ok ? "PASS" : "FAIL");

    int64_t time = 0;
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      scatterAdd_kernel(
        index,
        source,
        output,
        batch_size,
        output_size,
        vector_dim
      );
    }

    auto end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernel1: %f (us)\n", (time * 1e-3f) / repeat);

    time = 0;
    start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      scatterAdd2_kernel(
        index,
        source,
        output,
        batch_size,
        output_size,
        vector_dim
      );
    }

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of kernel2: %f (us)\n", (time * 1e-3f) / repeat);
  }


  free(source);
  free(index);
  free(output);
  free(output_ref);
}

int main(int argc, char* argv[])
{
  if (argc != 5) {
    printf("Usage: %s <batch size> <output size> <vector dimension> <repeat>\n", argv[0]);
    return 1;
  }
  const int batch_size = atoi(argv[1]);
  const int output_size = atoi(argv[2]);
  const int vector_dim = atoi(argv[3]);
  const int repeat = atoi(argv[4]);

  printf("batch_size: %d\n", batch_size);
  printf("output_size (range of index values): %d\n", output_size);
  printf("vector_dimension: %d\n", vector_dim);

  index_accumulate (batch_size, output_size, vector_dim, repeat) ;
  if (!ok) return 1;
  return 0;
}
