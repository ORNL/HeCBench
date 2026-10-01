#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <omp.h>
#include "reference.h"

#define threadsPerBlock  256

// Capacity of the per-team histogram. OpenMP has no way to size a team-local
// array at run time, so this is fixed when the kernel is built; the usable
// limit is the smaller of this and what the device actually offers.
// It is kept below a device's full 64 KB local memory so that the compiler's
// own per-kernel local-memory overhead still fits under the hardware limit.
#define sharedMemoryCapacity  (63 * 1024)

// Largest per-team local memory the device provides. OpenMP has no portable
// query for this; ompx_get_device_info is an Intel extension that also needs
// the offload runtime, so the Makefile only defines HAVE_OMPX_DEVICE_INFO for
// an offload build. Everything else falls back to the built-in capacity.
static int getDeviceLocalMemSize()
{
#ifdef HAVE_OMPX_DEVICE_INFO
  size_t localMemSize = 0, sizeRet = 0;
  if (ompx_get_device_info(omp_get_default_device(), ompx_devinfo_local_mem_size,
                           sizeof(localMemSize), &localMemSize, &sizeRet) == 0)
    return (int) localMemSize;
#endif
  return sharedMemoryCapacity;
}

#pragma omp declare target
template <typename input_t, typename IndexType>
static IndexType
getBin(input_t v, input_t minvalue, input_t maxvalue, IndexType nbins)
{
  IndexType bin = (v - minvalue) * nbins / (maxvalue - minvalue);
  // while each bin is inclusive at the lower end and exclusive at the higher,
  // i.e. [start, end) the last bin is inclusive at both, i.e. [start, end], in
  // order to include maxvalue if exists therefore when bin == nbins, adjust bin
  // to the last bin
  if (bin == nbins) bin--;
  return bin;
}
#pragma omp end declare target

/*
  Calculate the frequency of the input values.
  The GPU offloaded kernel atomically updates the global histogram tensor.
*/
template <typename output_t, typename input_t, typename IndexType>
void eval(IndexType input_size, int repeat)
{
  size_t input_size_bytes = sizeof(input_t) * input_size;

  input_t *input = (input_t*) malloc (input_size_bytes);

  // https://cplusplus.com/reference/random/normal_distribution/
  std::default_random_engine generator (123);
  std::normal_distribution<input_t> distribution(5.0,2.0);
  for (int i = 0; i < input_size; i++) {
    input[i] = distribution(generator);
  }

  auto min_iter = std::min_element(input, input+input_size);
  auto max_iter = std::max_element(input, input+input_size);

  input_t input_minvalue = *min_iter;
  input_t input_maxvalue = *max_iter;
  printf("Input min, max values: (%f %f)\n", (float)input_minvalue, (float)input_maxvalue);

  #pragma omp target enter data map(to: input[0:input_size])

  const int maxSharedMemory =
    std::min(getDeviceLocalMemSize(), (int)sharedMemoryCapacity);
  printf("Maximum shared local memory size per block in bytes: %d\n", maxSharedMemory);

  for (IndexType nbins = 768; nbins <= 768 * 32; nbins = nbins * 2) {

    printf("\nNumber of bins: %d\n", nbins);
    IndexType sharedMem = nbins * sizeof(output_t);

    IndexType output_size = nbins;
    size_t output_size_bytes = sizeof(output_t) * output_size;
    output_t *output = (output_t*) malloc (output_size_bytes);

    // reference
    output_t *output_r = (output_t*) calloc (output_size, sizeof(output_t));
    reference<output_t, input_t, IndexType>(
      output_r, input, nbins, input_minvalue, input_maxvalue,
      input_size, output_size, repeat);

    #pragma omp target enter data map(alloc: output[0:output_size])

    input_t minvalue = input_minvalue;
    input_t maxvalue = input_maxvalue;

    // determine memory type to use in the kernel
    printf("bincount using global atomics\n");

    #pragma omp target teams distribute parallel for
    for (IndexType i = 0; i < output_size; i++) output[i] = 0;

    auto start = std::chrono::steady_clock::now();
    for (int n = 0; n < repeat; n++) {
      #pragma omp target teams distribute parallel for
      for (IndexType linearIndex = 0; linearIndex < input_size; linearIndex++) {
        const input_t v = input[linearIndex];
        if (v >= minvalue && v <= maxvalue) {
          const IndexType bin = getBin<input_t, IndexType>(
                                v, minvalue, maxvalue, nbins);
          #pragma omp atomic update
          output[bin] += 1;
        }
      }
    }
    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average execution time of bincount kernel: %f (us)\n",
           (time * 1e-3f) / repeat);

    #pragma omp target update from(output[0:output_size])

    int status = memcmp(output, output_r, output_size_bytes);
    printf("%s\n", status ? "FAIL" : "PASS");

    if (sharedMem <= maxSharedMemory) {
      printf("\n");
      printf("bincount using global and local atomics\n");

      // number of teams mirrors the CUDA grid dimension
      const IndexType numTeams =
        (input_size + threadsPerBlock - 1) / threadsPerBlock;

      #pragma omp target teams distribute parallel for
      for (IndexType i = 0; i < output_size; i++) output[i] = 0;

      start = std::chrono::steady_clock::now();
      for (int n = 0; n < repeat; n++) {
        #pragma omp target teams num_teams(numTeams) thread_limit(threadsPerBlock)
        {
          // Per-team histogram. Declaring it in the teams region makes it
          // shared by the team's threads and lets the compiler place it in
          // team-local memory (LDS/SLM).
          output_t smem[sharedMemoryCapacity / sizeof(output_t)];

          #pragma omp parallel
          {
            const int nthreads = omp_get_num_threads();
            const int tid = omp_get_thread_num();
            const int team = omp_get_team_num();
            const int nteams = omp_get_num_teams();

            // zero the shared histogram
            for (IndexType i = tid; i < nbins; i += nthreads) smem[i] = 0;

            #pragma omp barrier

            // atomically accumulate into the shared histogram
            for (IndexType linearIndex = (IndexType)team * nthreads + tid;
                 linearIndex < input_size;
                 linearIndex += (IndexType)nteams * nthreads) {
              const input_t v = input[linearIndex];
              if (v >= minvalue && v <= maxvalue) {
                const IndexType bin = getBin<input_t, IndexType>(
                                      v, minvalue, maxvalue, nbins);
                #pragma omp atomic update
                smem[bin] += 1;
              }
            }

            #pragma omp barrier

            // flush the shared histogram to the global output
            for (IndexType i = tid; i < nbins; i += nthreads) {
              #pragma omp atomic update
              output[i] += smem[i];
            }
          }
        }
      }
      end = std::chrono::steady_clock::now();
      time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      printf("Average execution time of bincount kernel: %f (us)\n",
             (time * 1e-3f) / repeat);

      #pragma omp target update from(output[0:output_size])

      int status = memcmp(output, output_r, output_size_bytes);
      printf("%s\n", status ? "FAIL" : "PASS");
    }

    #pragma omp target exit data map(delete: output[0:output_size])
    free(output);
    free(output_r);
  }

  #pragma omp target exit data map(delete: input[0:input_size])
  free(input);
}

int main(int argc, char* argv[])
{
  if (argc != 3) {
    printf("Usage: %s <number of elements> <repeat>\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  eval<int, float, int>(n, repeat);

  return 0;
}
