#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <nccl.h>
#include <mpi.h>

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define GPUCHECK(cmd) do {                          \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",       \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


// map a host/device data type to its NCCL descriptor
template <typename T> struct ccl_traits;

template <> struct ccl_traits<float> {
  static constexpr ncclDataType_t dtype = ncclFloat;
  static const char* name() { return "float32"; }
  static float to_float(float v) { return v; }
  static float from_float(float v) { return v; }
};

template <> struct ccl_traits<__nv_bfloat16> {
  static constexpr ncclDataType_t dtype = ncclBfloat16;
  static const char* name() { return "bfloat16"; }
  static float to_float(__nv_bfloat16 v) { return __bfloat162float(v); }
  static __nv_bfloat16 from_float(float v) { return __float2bfloat16(v); }
};

template <typename T>
void run(ncclComm_t comm, cudaStream_t s, int mpi_rank, int mpi_size, int repeat)
{
  if (mpi_rank == 0)
    printf("\n=== Data type: %s ===\n", ccl_traits<T>::name());

  T *sendbuff, *recvbuff;
  T *h_sendbuff, *h_recvbuff;
  double start_time, stop_time, elapsed_time;

  for (int size = 1024*1024; size <= 1000 * 1024 * 1024; size = size * 10) {

    h_sendbuff = (T*) malloc (size * sizeof(T));
    for (int i = 0; i < size; i++) h_sendbuff[i] = ccl_traits<T>::from_float(1.f);

    h_recvbuff = (T*) malloc (size * sizeof(T));

    GPUCHECK(cudaMalloc(&sendbuff, size * sizeof(T)));
    GPUCHECK(cudaMemcpy(sendbuff, h_sendbuff, size * sizeof(T), cudaMemcpyHostToDevice));
    GPUCHECK(cudaMalloc(&recvbuff, size * sizeof(T)));

    start_time = MPI_Wtime();

    //communicating using NCCL
    for (int i = 0; i < repeat; i++) {
      NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size,
                              ccl_traits<T>::dtype, ncclSum, comm, s));
    }

    //completing NCCL operation by synchronizing on the CUDA stream
    GPUCHECK(cudaStreamSynchronize(s));
    stop_time = MPI_Wtime();
    elapsed_time = stop_time - start_time;

    if (mpi_rank == 0) {
      long int num_B = sizeof(T) * size * mpi_size;
      long int B_in_GB = 1 << 30;
      double num_GB = (double)num_B / (double)B_in_GB;
      double avg_time_per_transfer = elapsed_time / repeat;

      printf("Transfer size (B): %10li, Average Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n",
             num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer );
    }

    GPUCHECK(cudaMemcpy(h_recvbuff, recvbuff, size * sizeof(T), cudaMemcpyDeviceToHost));

    // CUDA cleanup
    GPUCHECK(cudaFree(sendbuff));
    GPUCHECK(cudaFree(recvbuff));

    bool ok = true;
    for (int i = 0; i < size; i++) {
      if (ccl_traits<T>::to_float(h_recvbuff[i]) != float(mpi_size)) {
         ok = false;
         break;
      }
    }
    free(h_sendbuff);
    free(h_recvbuff);

    printf("MPI Rank %d: %s\n", mpi_rank, ok ? "PASS" : "FAIL");
  }
}

int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  int num_gpus = 0;
  GPUCHECK(cudaGetDeviceCount(&num_gpus));

  if (num_gpus == 0) {
    fprintf(stderr, "ERROR: No GPU devices found on this node!\n");
    exit(EXIT_FAILURE);
  }

  int mpi_rank, mpi_size, local_rank;

  //initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));

  MPI_Comm local_comm;
  MPICHECK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED,
                               mpi_rank, MPI_INFO_NULL, &local_comm));
  MPICHECK(MPI_Comm_rank(local_comm, &local_rank));
  MPICHECK(MPI_Comm_free(&local_comm));

  if (local_rank >= num_gpus) {
    fprintf(stderr,
            "ERROR: Process %d needs GPU %d but only %d devices available\n",
            mpi_rank, local_rank, num_gpus);
    exit(EXIT_FAILURE);
  }

  GPUCHECK(cudaSetDevice(local_rank));

  cudaStream_t s;
  GPUCHECK(cudaStreamCreate(&s));

  printf("  MPI rank %d assigned to GPU device %d\n", mpi_rank, local_rank);

  ncclComm_t comm;
  ncclUniqueId id;

  //get NCCL unique ID at rank 0 and broadcast it to all others
  if (mpi_rank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  // each process joins the distributed NCCL communicator
  NCCLCHECK(ncclCommInitRank(&comm, mpi_size, id, mpi_rank));

  run<float>(comm, s, mpi_rank, mpi_size, repeat);
  run<__nv_bfloat16>(comm, s, mpi_rank, mpi_size, repeat);

  // NCCL cleanup
  NCCLCHECK(ncclCommFinalize(comm));
  NCCLCHECK(ncclCommDestroy(comm));

  // CUDA cleanup
  GPUCHECK(cudaStreamDestroy(s));

  //finalizing MPI
  MPICHECK(MPI_Finalize());

  return 0;
}
