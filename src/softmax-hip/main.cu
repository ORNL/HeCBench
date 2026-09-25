#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <hip/hip_runtime.h>
#include <hip/hip_cooperative_groups.h>
#ifdef HECBENCH_ENABLE_MPI_REPLICAS
#include <mpi.h>
#endif

#define BLOCK_SIZE 256

#define CHECK_HIP_ERROR(status)                                               \
  do {                                                                        \
    hipError_t error = (status);                                              \
    if (error != hipSuccess) {                                                \
      fprintf(stderr, "HIP error: '%s'(%d) at %s:%d\n",                      \
              hipGetErrorString(error), error, __FILE__, __LINE__);           \
      /* Avoid leaving peers waiting in the final correctness reduction. */   \
      /* MPI_Abort is only available in replica builds. */                    \
      fatal_exit();                                                           \
    }                                                                         \
  } while (0)

static void fatal_exit() {
#ifdef HECBENCH_ENABLE_MPI_REPLICAS
  MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
#endif
  exit(EXIT_FAILURE);
}


// A C model derived from the OpenCL kernel
void softMax_cpu(const int numSlice, const int sliceSize, const float* src, float* dest) {
  for (int i = 0; i < numSlice; i++) {
    float max_ = src[i * sliceSize];
    for (int j = 0; j < sliceSize; j++) {
      max_ = (max_ < src[i * sliceSize + j]) ? src[i * sliceSize + j] : max_;
    }
    float sum = 0;
    for (int j = 0; j < sliceSize; j++) {
      float e = expf(src[i * sliceSize + j] - max_);
      sum += e;
      dest[i * sliceSize + j] = e;
    }
    for (int j = 0; j < sliceSize; j++) {
      dest[i * sliceSize + j] /= sum;
    }
  }
}

__global__
void softMax (const int numSlice, const int sliceSize,
              const float* src, float* dest)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= numSlice) return;
  float max_ = src[i * sliceSize];
  for (int j = 0; j < sliceSize; j++) {
    max_ = max(max_, src[i * sliceSize + j]);
  }
  float sum = 0;
  for (int j = 0; j < sliceSize; j++) {
    sum += expf(src[i * sliceSize + j] - max_);
  }
  for (int j = 0; j < sliceSize; j++) {
    dest[i * sliceSize + j] = expf(src[i * sliceSize + j] - max_) / sum;
  }
}

__global__
void softMax2 (const int numSlice, const int sliceSize,
              const float* src, float* dest)
{
#if defined(__GFX8__) || defined(__GFX9__)
  #define WarpSize 64
#else
  #define WarpSize 32
#endif
  namespace cg = cooperative_groups;
  cg::thread_block block = cg::this_thread_block();
  cg::thread_block_tile<WarpSize> warp = cg::tiled_partition<WarpSize>(block);
  int i = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
  if (i >= numSlice) return;
  float max_ = src[i * sliceSize];
  for (int j = warp.thread_rank(); j < sliceSize; j += warp.size()) {
    max_ = max(max_, src[i * sliceSize + j]);
  }
  for (int offset = WarpSize/2; offset > 0; offset /= 2) {
      max_ = max(max_, warp.shfl_xor(max_, offset));
  }
  float sum = 0;
  for (int j = warp.thread_rank(); j < sliceSize; j += warp.size()) {
    sum += expf(src[i * sliceSize + j] - max_);
  }
  for (int offset = WarpSize/2; offset > 0; offset /= 2) {
      sum += warp.shfl_xor(sum, offset);
  }
  for (int j = warp.thread_rank(); j < sliceSize; j += warp.size()) {
    dest[i * sliceSize + j] = expf(src[i * sliceSize + j] - max_) / sum;
  }
}


int main(int argc, char* argv[]) {
#ifdef HECBENCH_ENABLE_MPI_REPLICAS
  MPI_Init(&argc, &argv);
  int rank, rank_count, local_rank;
  MPI_Comm local_comm;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &rank_count);
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank,
                      MPI_INFO_NULL, &local_comm);
  MPI_Comm_rank(local_comm, &local_rank);

  int visible_devices = 0;
  if (hipGetDeviceCount(&visible_devices) != hipSuccess || visible_devices < 1) {
    fprintf(stderr, "[rank %d] No visible HIP device\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  int selected_device = visible_devices == 1 ? 0 : local_rank;
  if (selected_device >= visible_devices) {
    fprintf(stderr,
            "[rank %d] local rank %d cannot be mapped to %d visible HIP devices\n",
            rank, local_rank, visible_devices);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  if (hipSetDevice(selected_device) != hipSuccess) {
    fprintf(stderr, "[rank %d] Failed to select HIP device %d\n", rank,
            selected_device);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
  char processor[MPI_MAX_PROCESSOR_NAME];
  int processor_length;
  MPI_Get_processor_name(processor, &processor_length);
  printf("[rank %d/%d local_rank %d] host=%s device=%d visible_devices=%d\n",
         rank, rank_count, local_rank, processor, selected_device,
         visible_devices);
  MPI_Comm_free(&local_comm);
#else
  int selected_device = 0;
#endif

  if (argc != 5) {
    printf("Usage: %s <number of slices> <slice size> <implementations> <repeat>\n", argv[0]);
    printf("implementation 0: naive\n");
    printf("implementation 1: optimized\n");
#ifdef HECBENCH_ENABLE_MPI_REPLICAS
    MPI_Finalize();
#endif
    return 1;
  }

  int numSlice = atoi(argv[1]);
  int sliceSize = atoi(argv[2]);
  int kernel = atoi(argv[3]);
  int repeat = atoi(argv[4]);
  int numElem = numSlice * sliceSize;

  float* input = (float*) aligned_alloc(1024, sizeof(float) * numElem);
  float* output_gpu = (float*) aligned_alloc(1024, sizeof(float) * numElem);
  float* output_cpu = (float*) aligned_alloc(1024, sizeof(float) * numElem);

  srand(2);
  for (int i = 0; i < numSlice; i++)
    for (int j = 0; j < sliceSize; j++)
      input[i*sliceSize+j] = rand() % 13;

  float *d_input, *d_output;
  CHECK_HIP_ERROR(hipMalloc((void**)&d_input, sizeof(float) * numElem));
  CHECK_HIP_ERROR(hipMalloc((void**)&d_output, sizeof(float) * numElem));
  CHECK_HIP_ERROR(
      hipMemcpy(d_input, input, sizeof(float) * numElem, hipMemcpyHostToDevice));

  if (kernel == 1) {
    int warp_size;
    CHECK_HIP_ERROR(hipDeviceGetAttribute(
        &warp_size, hipDeviceAttributeWarpSize, selected_device));
    dim3 grids ((numSlice+BLOCK_SIZE/warp_size-1)/(BLOCK_SIZE/warp_size));
    dim3 blocks (BLOCK_SIZE);

    CHECK_HIP_ERROR(hipDeviceSynchronize());
    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
      softMax2<<<grids, blocks>>>(numSlice, sliceSize, d_input, d_output);
    }

    CHECK_HIP_ERROR(hipDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
#ifdef HECBENCH_ENABLE_MPI_REPLICAS
    printf("[rank %d] Average kernel execution time: %f (ms)\n", rank,
           (time * 1e-6f) / repeat);
#else
    printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);
#endif
  }
  else {
    dim3 grids ((numSlice+BLOCK_SIZE-1)/BLOCK_SIZE);
    dim3 blocks (BLOCK_SIZE);

    CHECK_HIP_ERROR(hipDeviceSynchronize());
    auto start = std::chrono::steady_clock::now();

    for (int n = 0; n < repeat; n++) {
      softMax<<<grids, blocks>>>(numSlice, sliceSize, d_input, d_output);
    }

    CHECK_HIP_ERROR(hipDeviceSynchronize());
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
#ifdef HECBENCH_ENABLE_MPI_REPLICAS
    printf("[rank %d] Average kernel execution time: %f (ms)\n", rank,
           (time * 1e-6f) / repeat);
#else
    printf("Average kernel execution time: %f (ms)\n", (time * 1e-6f) / repeat);
#endif
  }

  CHECK_HIP_ERROR(hipMemcpy(output_gpu, d_output, sizeof(float) * numElem,
                            hipMemcpyDeviceToHost));

  // verification
  bool ok = true;
  softMax_cpu(numSlice, sliceSize, input, output_cpu);
  for (int i = 0; i < numElem; i++) {
    if (fabsf(output_cpu[i] - output_gpu[i]) > 1e-3) {
      printf("@index %d host: %f device: %f\n", i, output_cpu[i], output_gpu[i]);
      ok = false;
      break;
    }
  }
#ifdef HECBENCH_ENABLE_MPI_REPLICAS
  printf("[rank %d] %s\n", rank, ok ? "PASS" : "FAIL");
  int local_pass = ok ? 1 : 0;
  int passed_count = 0;
  MPI_Allreduce(&local_pass, &passed_count, 1, MPI_INT, MPI_SUM,
                MPI_COMM_WORLD);
  if (rank == 0) {
    printf("OVERALL %s (%d/%d replica ranks)\n",
           passed_count == rank_count ? "PASS" : "FAIL", passed_count,
           rank_count);
  }
#else
  printf("%s\n", ok ? "PASS" : "FAIL");
#endif

  free(input);
  free(output_cpu);
  free(output_gpu);
  CHECK_HIP_ERROR(hipFree(d_input));
  CHECK_HIP_ERROR(hipFree(d_output));
#ifdef HECBENCH_ENABLE_MPI_REPLICAS
  MPI_Finalize();
  return passed_count == rank_count ? 0 : 2;
#endif
  return 0;
}
