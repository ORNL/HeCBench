#include <stdio.h>
#include <unistd.h>
#include <stdint.h>
#include <stdlib.h>
#include <sycl/sycl.hpp>
#include <oneapi/ccl.hpp>
#include <mpi.h>

#define MPICHECK(cmd) do {                          \
  int e = cmd;                                      \
  if( e != MPI_SUCCESS ) {                          \
    printf("Failed: MPI error %s:%d '%d'\n",        \
        __FILE__,__LINE__, e);   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)

void mpi_finalize() {
  int is_finalized = 0;
  MPI_Finalized(&is_finalized);
  if (!is_finalized) MPI_Finalize();
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  // level-zero gpu
  auto const& gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  int num_gpus = gpu_devices.size();

  if (num_gpus == 0) {
    fprintf(stderr, "ERROR: No GPU devices found on this node!\n");
    exit(EXIT_FAILURE);
  }

  int mpi_rank, mpi_size, local_rank;

  ccl::init();

  //initializing MPI
  MPICHECK(MPI_Init(&argc, &argv));
  MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank));
  MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &mpi_size));

  atexit(mpi_finalize);

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

  // create kvs at rank 0 and broadcast its address to all others
  ccl::shared_ptr_class<ccl::kvs> kvs;
  ccl::kvs::address_type main_addr;
  if (mpi_rank == 0) {
    kvs = ccl::create_main_kvs();
    main_addr = kvs->get_address();
    MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
  }
  else {
    MPI_Bcast((void *)main_addr.data(), main_addr.size(), MPI_BYTE, 0, MPI_COMM_WORLD);
    kvs = ccl::create_kvs(main_addr);
  }

  //picking a GPU based on local_rank, allocate device buffers
  auto q = sycl::queue(gpu_devices[local_rank], sycl::property::queue::in_order());

  // create communicator
  auto dev = ccl::create_device(q.get_device());
  auto ctx = ccl::create_context(q.get_context());
  auto comm = ccl::create_communicator(mpi_size, mpi_rank, dev, ctx, kvs);

  // create stream
  auto stream = ccl::create_stream(q);

  float *sendbuff, *recvbuff;
  float *h_sendbuff, *h_recvbuff;
  double start_time, stop_time, elapsed_time;

  for (int size = 1024*1024; size <= 1000 * 1024 * 1024; size = size * 10) {

    h_sendbuff = (float*) malloc (size * sizeof(float));
    for (int i = 0; i < size; i++) h_sendbuff[i] = 1;

    h_recvbuff = (float*) malloc (size * sizeof(float));

    sendbuff = sycl::malloc_device<float>(size, q);
    q.memcpy(sendbuff, h_sendbuff, size * sizeof(float));
    recvbuff = sycl::malloc_device<float>(size, q);

    start_time = MPI_Wtime();

    for (int i = 0; i < repeat; i++) {
      ccl::allreduce(sendbuff, recvbuff, size, ccl::datatype::float32, ccl::reduction::sum, comm, stream);
    }

    q.wait();

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

    q.memcpy(h_recvbuff, recvbuff, size * sizeof(float)).wait();

    //free device buffers
    sycl::free(sendbuff, q);
    sycl::free(recvbuff, q);

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

  return 0;
}
