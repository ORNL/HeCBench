#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>

#include <hip/hip_runtime.h>
#include <rccl/rccl.h>
#include <mpi.h>

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e); \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define GPUCHECK(cmd) do {                         \
  hipError_t e = cmd;                              \
  if( e != hipSuccess ) {                          \
    printf("Failed: Hip error %s:%d '%s'\n",       \
        __FILE__,__LINE__,hipGetErrorString(e));   \
    exit(EXIT_FAILURE);                            \
  }                                                \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",       \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


int main(int argc, char* argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  int num_gpus = 0;
  GPUCHECK(hipGetDeviceCount(&num_gpus));

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

  GPUCHECK(hipSetDevice(local_rank));

  hipStream_t s;
  GPUCHECK(hipStreamCreate(&s));

  printf("  MPI rank %d assigned to GPU device %d\n", mpi_rank, local_rank);

  ncclComm_t comm;
  ncclUniqueId id;

  //get NCCL unique ID at rank 0 and broadcast it to all others
  if (mpi_rank == 0) ncclGetUniqueId(&id);
  MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

  // each process joins the distributed NCCL communicator
  NCCLCHECK(ncclCommInitRank(&comm, mpi_size, id, mpi_rank));

  float *sendbuff, *recvbuff;
  float *h_sendbuff, *h_recvbuff;
  double start_time, stop_time, elapsed_time;

  for (int size = 1024*1024; size <= 1000 * 1024 * 1024; size = size * 10) {

    h_sendbuff = (float*) malloc (size * sizeof(float));
    for (int i = 0; i < size; i++) h_sendbuff[i] = 1;

    h_recvbuff = (float*) malloc (size * sizeof(float));

    GPUCHECK(hipMalloc(&sendbuff, size * sizeof(float)));
    GPUCHECK(hipMemcpy(sendbuff, h_sendbuff, size * sizeof(float), hipMemcpyHostToDevice));
    GPUCHECK(hipMalloc(&recvbuff, size * sizeof(float)));

    start_time = MPI_Wtime();

    //communicating using NCCL
    for (int i = 0; i < repeat; i++) {
      NCCLCHECK(ncclAllReduce((const void*)sendbuff, (void*)recvbuff, size,
                              ncclFloat, ncclSum, comm, s));
    }

    //completing NCCL operation by synchronizing on the HIP stream
    GPUCHECK(hipStreamSynchronize(s));
    stop_time = MPI_Wtime();
    elapsed_time = stop_time - start_time;

    if (mpi_rank == 0) {
      long int num_B = sizeof(float) * size * mpi_size;
      long int B_in_GB = 1 << 30;
      double num_GB = (double)num_B / (double)B_in_GB;
      double avg_time_per_transfer = elapsed_time / repeat;

      printf("Transfer size (B): %10li, Average Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n",
             num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer );
    }

    GPUCHECK(hipMemcpy(h_recvbuff, recvbuff, size * sizeof(float), hipMemcpyDeviceToHost));

    // HIP cleanup
    GPUCHECK(hipFree(sendbuff));
    GPUCHECK(hipFree(recvbuff));

    bool ok = true;
    for (int i = 0; i < size; i++) {
      if (h_recvbuff[i] != float(mpi_size)) {
         ok = false;
         break;
      }
    }
    free(h_sendbuff);
    free(h_recvbuff);

    printf("MPI Rank %d: %s\n", mpi_rank, ok ? "PASS" : "FAIL");
  }

  // NCCL cleanup
  NCCLCHECK(ncclCommFinalize(comm));
  NCCLCHECK(ncclCommDestroy(comm));

  // HIP cleanup
  GPUCHECK(hipStreamDestroy(s));

  //finalizing MPI
  MPICHECK(MPI_Finalize());

  return 0;
}
