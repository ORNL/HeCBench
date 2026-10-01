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

using bfloat16 = sycl::ext::oneapi::bfloat16;

// map a host/device data type to its oneCCL descriptor
template <typename T> struct ccl_traits;

template <> struct ccl_traits<float> {
  static constexpr ccl::datatype dtype = ccl::datatype::float32;
  static const char* name() { return "float32"; }
  static float to_float(float v) { return v; }
  static float from_float(float v) { return v; }
};

template <> struct ccl_traits<bfloat16> {
  static constexpr ccl::datatype dtype = ccl::datatype::bfloat16;
  static const char* name() { return "bfloat16"; }
  static float to_float(bfloat16 v) { return static_cast<float>(v); }
  static bfloat16 from_float(float v) { return bfloat16(v); }
};

template <typename T, typename Comm, typename Stream>
void run(sycl::queue &q, Comm &comm, Stream &stream,
         int mpi_rank, int mpi_size, int repeat)
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

    sendbuff = sycl::malloc_device<T>(size, q);
    q.memcpy(sendbuff, h_sendbuff, size * sizeof(T));
    recvbuff = sycl::malloc_device<T>(size, q);

    start_time = MPI_Wtime();

    for (int i = 0; i < repeat; i++) {
      ccl::allreduce(sendbuff, recvbuff, size, ccl_traits<T>::dtype,
                     ccl::reduction::sum, comm, stream);
    }

    q.wait();

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

    q.memcpy(h_recvbuff, recvbuff, size * sizeof(T)).wait();

    //free device buffers
    sycl::free(sendbuff, q);
    sycl::free(recvbuff, q);

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

  run<float>(q, comm, stream, mpi_rank, mpi_size, repeat);
  run<bfloat16>(q, comm, stream, mpi_rank, mpi_size, repeat);

  return 0;
}
