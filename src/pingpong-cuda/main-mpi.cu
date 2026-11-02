#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <mpi.h>
#include "gpu_aware_mpi.h"

// Macro for checking errors in CUDA API calls
#define cudaErrorCheck(call)                                                              \
  do{                                                                                     \
    cudaError_t cuErr = call;                                                             \
    if(cudaSuccess != cuErr){                                                             \
      printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(cuErr));\
      fflush(stdout);                                                                     \
      MPI_Abort(MPI_COMM_WORLD, 1);                                                       \
      exit(1);                                                                            \
    }                                                                                     \
  }while(0)


__global__
void test(double *d, const long int n) {
  for (long i = blockDim.x * blockIdx.x + threadIdx.x;
       i < n; i += blockDim.x * gridDim.x) {
    d[i] = d[i] + 1;
  }
}

int main(int argc, char *argv[])
{
  /* -------------------------------------------------------------------------------------------
     MPI Initialization 
     --------------------------------------------------------------------------------------------*/
  MPI_Init(&argc, &argv);

  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  MPI_Status stat;

  if(size != 2){
    if(rank == 0){
      printf("This program requires exactly 2 MPI ranks, but you are attempting to use %d! Exiting...\n", size);
    }
    MPI_Finalize();
    exit(0);
  }

  // Map MPI ranks to GPUs
  int num_devices = 0;
  cudaErrorCheck( cudaGetDeviceCount(&num_devices) );
  cudaErrorCheck( cudaSetDevice(rank % num_devices) );

  gpu_aware_mpi_require(GPU_AWARE_MPI_KIND_CUDA, rank);

  //   Loop from 512 KiB to 1 GB (8 * 2^i bytes, i = 16..27)
  for(int i=16; i<=27; i++){

    long int N = 1 << i;

    double *h_A, *d_A;
    h_A = (double*) malloc (N*sizeof(double)); 
    cudaErrorCheck( cudaMalloc((void**)&d_A, N*sizeof(double)) );
    cudaErrorCheck( cudaMemset(d_A, 0, N*sizeof(double)) );
    cudaErrorCheck( cudaDeviceSynchronize() );

    const int tag1 = 10;
    const int tag2 = 20;

    int loop_count = 50;

    // Warm-up and validate MPI pingpong
    for(int i=1; i<=5; i++){
      if(rank == 0){
        MPI_Send(d_A, N, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD);
        MPI_Recv(d_A, N, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD, &stat);
      }
      else if(rank == 1){
        MPI_Recv(d_A, N, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, &stat);
        test<<<1024, 256>>>(d_A, N);
        cudaErrorCheck( cudaDeviceSynchronize() );
        MPI_Send(d_A, N, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
      }
    }
    cudaErrorCheck(cudaMemcpy(h_A, d_A, N*sizeof(double), cudaMemcpyDeviceToHost));
    int valid = 1;
    for (long int j = 0; j < N; j++) {
      if(h_A[j] != 5) {
        fprintf(stderr,
                "ERROR: rank %d: MPI pingpong validation failed at N=%ld index %ld value %.17g\n",
                rank, N, j, h_A[j]);
        fflush(stderr);
        valid = 0;
        break;
      }
    }
    int valid_all = 0;
    MPI_Allreduce(&valid, &valid_all, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD);
    if (!valid_all)
      MPI_Abort(MPI_COMM_WORLD, 1);

    free(h_A);

    // Time loop_count iterations of data transfer size 8*N bytes
    double start_time, stop_time, elapsed_time;
    start_time = MPI_Wtime();

    for(int i=1; i<=loop_count; i++){
      if(rank == 0){
        MPI_Send(d_A, N, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD);
        MPI_Recv(d_A, N, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD, &stat);
      }
      else if(rank == 1){
        MPI_Recv(d_A, N, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, &stat);
        MPI_Send(d_A, N, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
      }
    }

    stop_time = MPI_Wtime();
    elapsed_time = stop_time - start_time;

    long int num_B = 8*N;
    double num_GB = (double)num_B / 1.0e9;
    double avg_time_per_transfer = elapsed_time / (2.0*(double)loop_count);

    if(rank == 0)
      printf("MPI : Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n",
             num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer );

    cudaErrorCheck( cudaFree(d_A) );
  }

  MPI_Finalize();

  return 0;
}
