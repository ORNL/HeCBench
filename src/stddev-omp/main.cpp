/*
 * Copyright (c) 2018-2020, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>
#include <omp.h>
#include "reference.h"

/**
 * @brief Compute stddev of the input matrix
 *
 * Stddev operation is assumed to be performed on a given column.
 *
 * @tparam Type the data type
 * @tparam IdxType Integer type used to for addressing
 * @param std the output stddev vector
 * @param data the input matrix
 * @param D number of columns of data
 * @param N number of rows of data
 * @param sample whether to evaluate sample stddev or not. In other words,
 * whether
 *  to normalize the output using N-1 or N, for true or false, respectively
 */
template <typename Type, typename IdxType = int>
void stddev(Type *std, const Type *data, IdxType D, IdxType N, bool sample) {
  static const int TPB = 256;
  static const int RowsPerThread = 4;
  static const int ColsPerBlk = 32;
  static const int RowsPerBlk = (TPB / ColsPerBlk) * RowsPerThread;

  static const int TeamX = (N + (IdxType)RowsPerBlk - 1) / (IdxType)RowsPerBlk;
  static const int TeamY = (D + (IdxType)ColsPerBlk - 1) / (IdxType)ColsPerBlk;
  static const int Teams = TeamX * TeamY;

  // required for atomics
  #pragma omp target teams distribute parallel for thread_limit(256)
  for (int i = 0; i < D; i++)
    std[i] = (Type)0;

  #pragma omp target teams num_teams(Teams) thread_limit(TPB)
  {
    Type sstd[ColsPerBlk];
    #pragma omp parallel
    {
      int tx = omp_get_thread_num();
      int bx = omp_get_team_num() % TeamX;
      int by = omp_get_team_num() / TeamX;
      int gridDim_x = TeamX;
 
      const int RowsPerBlkPerIter = TPB / ColsPerBlk;
      IdxType thisColId = tx % ColsPerBlk;
      IdxType thisRowId = tx / ColsPerBlk;
      IdxType colId = thisColId + ((IdxType)by * ColsPerBlk);
      IdxType rowId = thisRowId + ((IdxType)bx * RowsPerBlkPerIter);
      Type thread_data = Type(0);
      const IdxType stride = RowsPerBlkPerIter * gridDim_x;
      for (IdxType i = rowId; i < N; i += stride) {
        Type val = (colId < D) ? data[i * D + colId] : Type(0);
        thread_data += val * val;
      }
      if (tx < ColsPerBlk) sstd[tx] = Type(0);
      #pragma omp barrier

      #pragma omp atomic update
      sstd[thisColId] += thread_data;

      #pragma omp barrier

      if (tx < ColsPerBlk) {
        #pragma omp atomic update
        std[colId] += sstd[thisColId];
      }
    }
  }

  IdxType sampleSize = sample ? N-1 : N;
  #pragma omp target teams distribute parallel for thread_limit(TPB)
  for (int i = 0; i < D; i++)
    std[i] = sqrtf(std[i] / sampleSize);
}

int main(int argc, char* argv[]) {
  if (argc != 4) {
    printf("Usage: %s <D> <N> <repeat>\n", argv[0]);
    printf("D: number of columns of data (must be a multiple of 32)\n");
    printf("N: number of rows of data (at least one row)\n");
    return 1;
  }
  int D = atoi(argv[1]); // columns must be a multiple of 32
  int N = atoi(argv[2]); // at least one row
  int repeat = atoi(argv[3]);

  bool sample = true;
  long inputSize = D * N;
  long inputSizeByte = inputSize * sizeof(float);
  float *data = (float*) malloc (inputSizeByte);

  // input data 
  srand(123);
  for (int i = 0; i < N; i++)
    for (int j = 0; j < D; j++) 
      data[i*D + j] = rand() / (float)RAND_MAX; 

  // host and device results
  long outputSize = D;
  long outputSizeByte = outputSize * sizeof(float);
  float *std  = (float*) malloc (outputSizeByte);
  float *std_ref  = (float*) malloc (outputSizeByte);

  #pragma omp target data map (to: data[0:inputSize]) map (from: std[0:outputSize])
  {
    // warmup
    stddev(std, data, D, N, sample);

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++)
      stddev(std, data, D, N, sample);

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of stddev kernels: %f (s)\n", (time * 1e-9f) / repeat);
  }

  // verify
  stddev_ref(std_ref, data, D, N, sample);

  bool ok = true;
  for (int i = 0; i < D; i++) {
    if (fabsf(std_ref[i] - std[i]) > 1e-3) {
      ok = false;
      break;
    }
  }

  printf("%s\n", ok ? "PASS" : "FAIL");
  free(std_ref);
  free(std);
  free(data);
  if (!ok) return 1;
  return 0;
}
