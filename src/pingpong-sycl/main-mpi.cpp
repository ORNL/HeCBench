#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <sycl/sycl.hpp>
#include <mpi.h>

void test(sycl::nd_item<1> item, double *d, const long int n) {
  for (long i = item.get_global_id(0);
       i < n; i += item.get_local_range(0) * item.get_group_range(0)) {
    d[i] = d[i] + 1;
  }
}

// Distinct values so a no-op MPI transfer cannot match by accident.
static const double kProbeA = 123456789.0;
static const double kProbeB = 987654321.0;
static const double kPoisonA = -1.0;
static const double kPoisonB = -2.0;
static const int kProbeTimeoutSec = 30;
static const long kTinyN = 2;
static const long kRendezN = 8192;      // 64 KiB; typical eager vs rendezvous boundary
static const long kFirstBenchN = 1 << 16;  // same element count as the first timed size

static std::atomic<int> g_probe_done{0};

static void gpu_aware_probe_watchdog() {
  const auto deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(kProbeTimeoutSec);
  while (!g_probe_done.load(std::memory_order_relaxed)) {
    if (std::chrono::steady_clock::now() >= deadline) {
      fprintf(stderr,
              "ERROR: GPU-aware MPI probe timed out (MPI likely hung on a device pointer).\n"
              "main-mpi requires an MPI that can send/recv GPU buffers.\n");
      fflush(stderr);
      _exit(1);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
}

static void probe_abort(const char *why)
{
  fprintf(stderr, "ERROR: %s\n", why);
  fflush(stderr);
  g_probe_done.store(1, std::memory_order_relaxed);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

static void mpi_or_abort(int err, const char *what)
{
  if (err == MPI_SUCCESS)
    return;
  char str[MPI_MAX_ERROR_STRING];
  int len = 0;
  MPI_Error_string(err, str, &len);
  fprintf(stderr, "ERROR: %s: %s\n", what, str);
  fflush(stderr);
  g_probe_done.store(1, std::memory_order_relaxed);
  MPI_Abort(MPI_COMM_WORLD, 1);
}

static int check_pair(int rank, const char *why, const double *h,
                      double e0, double e1)
{
  if (h[0] == e0 && h[1] == e1)
    return 1;
  fprintf(stderr,
          "ERROR: rank %d: GPU-aware MPI probe failed (%s): "
          "got [%.17g, %.17g] expected [%.17g, %.17g]\n"
          "main-mpi requires an MPI that can send/recv GPU buffers.\n",
          rank, why, h[0], h[1], e0, e1);
  fflush(stderr);
  return 0;
}

static int check_all(int rank, const char *why, const double *h, long n,
                     double expected)
{
  for (long i = 0; i < n; i++) {
    if (h[i] != expected) {
      fprintf(stderr,
              "ERROR: rank %d: GPU-aware MPI probe failed (%s): "
              "d[%ld]=%.17g expected %.17g (n=%ld)\n"
              "main-mpi requires an MPI that can send/recv GPU buffers.\n",
              rank, why, i, h[i], expected, n);
      fflush(stderr);
      return 0;
    }
  }
  return 1;
}

static void launch_inc(sycl::queue &q, double *d, long ninc)
{
  const size_t gws = (ninc <= 1) ? 1 : (size_t)1024 * 256;
  const size_t lws = (ninc <= 1) ? 1 : 256;
  q.submit([&] (sycl::handler &cgh) {
    cgh.parallel_for(sycl::nd_range<1>(gws, lws), [=] (sycl::nd_item<1> item) {
      test(item, d, ninc);
    });
  }).wait();
}

// MPI_SUCCESS is not enough: copy device memory back and check payload plus a
// device-side increment. Tiny messages can take an eager/host-staging path, so
// also probe 64 KiB and the first timed size. Host MPI of a flag is used only
// to agree on the result after those checks.
static int probe_roundtrip(sycl::queue &q, int rank, long n, int inc_all, int tag)
{
  MPI_Status stat;
  double *h_payload = (double *)malloc((size_t)n * sizeof(double));
  double *h_poison = (double *)malloc((size_t)n * sizeof(double));
  double *h_dev = (double *)malloc((size_t)n * sizeof(double));
  if (!h_payload || !h_poison || !h_dev)
    probe_abort("GPU-aware MPI probe: host malloc failed");

  if (n == kTinyN) {
    h_payload[0] = kProbeA;
    h_payload[1] = kProbeB;
    h_poison[0] = kPoisonA;
    h_poison[1] = kPoisonB;
  } else {
    for (long i = 0; i < n; i++) {
      h_payload[i] = kProbeA;
      h_poison[i] = kPoisonA;
    }
  }

  double *d = sycl::malloc_device<double>((size_t)n, q);
  if (!d)
    probe_abort("GPU-aware MPI probe: sycl::malloc_device failed");
  q.memcpy(d, h_poison, (size_t)n * sizeof(double)).wait();
  mpi_or_abort(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");

  int ok = 1;
  if (rank == 0) {
    q.memcpy(d, h_payload, (size_t)n * sizeof(double)).wait();
    mpi_or_abort(MPI_Send(d, (int)n, MPI_DOUBLE, 1, tag, MPI_COMM_WORLD),
                 "MPI_Send");
    mpi_or_abort(MPI_Recv(d, (int)n, MPI_DOUBLE, 1, tag, MPI_COMM_WORLD, &stat),
                 "MPI_Recv");
    q.wait();
    q.memcpy(h_dev, d, (size_t)n * sizeof(double)).wait();
    if (n == kTinyN)
      ok = check_pair(rank, "rank 0 after round-trip (expect kernel +1 on elt 0)",
                      h_dev, kProbeA + 1.0, kProbeB);
    else
      ok = check_all(rank, "rank 0 after round-trip (expect kernel +1)",
                     h_dev, n, kProbeA + 1.0);
  } else {
    mpi_or_abort(MPI_Recv(d, (int)n, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD, &stat),
                 "MPI_Recv");
    q.wait();
    q.memcpy(h_dev, d, (size_t)n * sizeof(double)).wait();
    if (n == kTinyN)
      ok = check_pair(rank, "rank 1 after Recv (poison must be overwritten on device)",
                      h_dev, kProbeA, kProbeB);
    else
      ok = check_all(rank, "rank 1 after Recv (poison must be overwritten on device)",
                     h_dev, n, kProbeA);
    launch_inc(q, d, inc_all ? n : 1);
    q.memcpy(h_dev, d, (size_t)n * sizeof(double)).wait();
    int ok_k;
    if (n == kTinyN)
      ok_k = check_pair(rank, "rank 1 after kernel (only elt 0 incremented)",
                        h_dev, kProbeA + 1.0, kProbeB);
    else
      ok_k = check_all(rank, "rank 1 after kernel (all elts incremented)",
                       h_dev, n, kProbeA + 1.0);
    ok = ok && ok_k;
    mpi_or_abort(MPI_Send(d, (int)n, MPI_DOUBLE, 0, tag, MPI_COMM_WORLD),
                 "MPI_Send");
  }

  sycl::free(d, q);
  free(h_payload);
  free(h_poison);
  free(h_dev);
  return ok;
}

static void probe_gpu_aware_mpi(sycl::queue &q, int rank, int use_watchdog)
{
  g_probe_done.store(0, std::memory_order_relaxed);
  std::thread watchdog;
  if (use_watchdog)
    watchdog = std::thread(gpu_aware_probe_watchdog);

  // Always run every size so the two ranks cannot skip a matching Send/Recv.
  const int ok_tiny = probe_roundtrip(q, rank, kTinyN, 0, 91);
  const int ok_rendez = probe_roundtrip(q, rank, kRendezN, 1, 92);
  const int ok_bench = probe_roundtrip(q, rank, kFirstBenchN, 1, 93);
  const int ok = ok_tiny && ok_rendez && ok_bench;

  int ok_all = 0;
  mpi_or_abort(MPI_Allreduce(&ok, &ok_all, 1, MPI_INT, MPI_LAND, MPI_COMM_WORLD),
               "MPI_Allreduce");

  g_probe_done.store(1, std::memory_order_relaxed);
  if (use_watchdog)
    watchdog.join();

  if (!ok_all)
    probe_abort("GPU-aware MPI probe failed: device buffers were not transferred "
                "or updated correctly");
}

int main(int argc, char *argv[])
{
  /* -------------------------------------------------------------------------------------------
     MPI Initialization
     --------------------------------------------------------------------------------------------*/
  int provided = 0;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

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

  const int use_watchdog = (provided >= MPI_THREAD_FUNNELED);
  if (!use_watchdog && rank == 0) {
    fprintf(stderr,
            "WARNING: MPI provided thread level %d < MPI_THREAD_FUNNELED (%d); "
            "GPU-aware hang watchdog disabled. Device-buffer checks still run.\n",
            provided, MPI_THREAD_FUNNELED);
    fflush(stderr);
  }

  auto const& gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  int num_devices = (int)gpu_devices.size();
  if (num_devices < 1)
    probe_abort("no SYCL GPU devices");
  sycl::queue q(gpu_devices[rank % num_devices], sycl::property::queue::in_order());

  probe_gpu_aware_mpi(q, rank, use_watchdog);

  //   Loop from 512 KiB to 1 GB (8 * 2^i bytes, i = 16..27)
  for(int i=16; i<=27; i++){

    long int N = 1 << i;

    double *h_A, *d_A;
    h_A = (double*) malloc (N*sizeof(double)); 
    d_A = sycl::malloc_device<double>(N, q);
    q.memset(d_A, 0, N*sizeof(double)).wait();

    const int tag1 = 10;
    const int tag2 = 20;

    int loop_count = 50;

    // Warm-up loop
    for(int i=1; i<=5; i++){
      if(rank == 0){
        MPI_Send(d_A, N, MPI_DOUBLE, 1, tag1, MPI_COMM_WORLD);
        MPI_Recv(d_A, N, MPI_DOUBLE, 1, tag2, MPI_COMM_WORLD, &stat);
        q.wait();
      }
      else if(rank == 1){
        MPI_Recv(d_A, N, MPI_DOUBLE, 0, tag1, MPI_COMM_WORLD, &stat);
        q.wait();
        q.submit([&] (sycl::handler &cgh) {
          cgh.parallel_for(
            sycl::nd_range<1>(1024*256, 256), [=] (sycl::nd_item<1> item) {
              test(item, d_A, N);
          });
        }).wait();
        MPI_Send(d_A, N, MPI_DOUBLE, 0, tag2, MPI_COMM_WORLD);
      }
    }
    q.memcpy(h_A, d_A, N*sizeof(double)).wait();
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

    // Time ping-pong for loop_count iterations of data transfer size 8*N bytes
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
      printf("MPI: Transfer size (B): %10li, Transfer Time (s): %15.9f, Bandwidth (GB/s): %15.9f\n",
             num_B, avg_time_per_transfer, num_GB/avg_time_per_transfer );

    sycl::free(d_A, q);
  }

  MPI_Finalize();

  return 0;
}
