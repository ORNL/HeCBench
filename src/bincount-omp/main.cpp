#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
#include <omp.h>
#include "reference.h"

#ifndef THREADS_PER_BLOCK
#define THREADS_PER_BLOCK  256
#endif

// Bound the auto-tuner's temporary storage and search space independently of
// the compiler or target architecture.
static constexpr size_t PARTIAL_HISTOGRAM_MEMORY_BUDGET =
  64u * 1024u * 1024u;
static constexpr int MAX_PARTIAL_HISTOGRAMS = 1024;
static constexpr int MAX_LOCAL_HISTOGRAM_TEAMS = 1024;
static constexpr int MAX_LOCAL_HISTOGRAM_BINS = 4096;
static constexpr int CALIBRATION_ITERATIONS = 3;

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
  Runtime calibration selects direct atomics, global partial histograms, or
  an OpenMP 4.5 team-scoped histogram. Selection depends on actual compiler
  and device behavior rather than compiler-name heuristics.
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

  #pragma omp target data map(to: input[0:input_size])
  {
  for (IndexType nbins = 768; nbins <= 768 * 32; nbins = nbins * 2) {

    printf("\nNumber of bins: %d\n", nbins);

    IndexType output_size = nbins;
    size_t output_size_bytes = sizeof(output_t) * output_size;
    output_t *output = (output_t*) malloc (output_size_bytes);

    // reference
    output_t *output_r = (output_t*) calloc (output_size, sizeof(output_t));
    reference<output_t, input_t, IndexType>(
      output_r, input, nbins, input_minvalue, input_maxvalue,
      input_size, output_size, repeat);

    input_t minvalue = input_minvalue;
    input_t maxvalue = input_maxvalue;

    const IndexType inputTeams =
      (input_size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    const size_t histogramBytes = (size_t)nbins * sizeof(output_t);
    const IndexType memoryLimitedHistograms =
      (IndexType)(PARTIAL_HISTOGRAM_MEMORY_BUDGET / histogramBytes);
    const IndexType maxPartialHistograms =
      std::min<IndexType>(
        inputTeams,
        std::min<IndexType>(MAX_PARTIAL_HISTOGRAMS,
                            memoryLimitedHistograms));
    const size_t partialSize =
      (size_t)maxPartialHistograms * (size_t)nbins;
    output_t *partial =
      (output_t*) malloc(partialSize * sizeof(output_t));

    std::vector<IndexType> candidates;
    candidates.push_back(0);
    if (maxPartialHistograms > 1) {
      for (IndexType count = 32; count < maxPartialHistograms; count *= 4)
        candidates.push_back(count);
      if (candidates.back() != maxPartialHistograms)
        candidates.push_back(maxPartialHistograms);
    }
    if (nbins <= MAX_LOCAL_HISTOGRAM_BINS) {
      const IndexType maxLocalTeams =
        std::min<IndexType>(inputTeams, MAX_LOCAL_HISTOGRAM_TEAMS);
      for (IndexType count = 128; count < maxLocalTeams; count *= 4)
        candidates.push_back(-count);
      if (maxLocalTeams > 0 &&
          (candidates.empty() || candidates.back() != -maxLocalTeams))
        candidates.push_back(-maxLocalTeams);
    }
    #pragma omp target data map(from: output[0:output_size]) \
                            map(alloc: partial[0:partialSize])
    {
      #pragma omp target teams distribute parallel for
      for (IndexType i = 0; i < output_size; i++) output[i] = 0;
      #pragma omp target teams distribute parallel for
      for (size_t i = 0; i < partialSize; i++) partial[i] = 0;

    long long bestTime = 0;
    IndexType bestConfiguration = 0;
    for (const IndexType activeConfiguration : candidates) {
      // The first trial warms this exact target region. Only the second trial
      // participates in selection, excluding JIT and runtime initialization.
      for (int trial = 0; trial < 2; trial++) {
        #pragma omp target teams distribute parallel for
        for (IndexType i = 0; i < output_size; i++) output[i] = 0;

        const auto calibrationStart = std::chrono::steady_clock::now();
        if (activeConfiguration == 0) {
          for (int iteration = 0;
               iteration < CALIBRATION_ITERATIONS; iteration++) {
            #pragma omp target teams distribute parallel for
            for (IndexType linearIndex = 0;
                 linearIndex < input_size; linearIndex++) {
              const input_t v = input[linearIndex];
              if (v >= minvalue && v <= maxvalue) {
                const IndexType bin = getBin<input_t, IndexType>(
                                      v, minvalue, maxvalue, nbins);
                #pragma omp atomic update
                output[bin] += 1;
              }
            }
          }
        } else if (activeConfiguration < 0) {
          const IndexType localTeams = -activeConfiguration;
          for (int iteration = 0;
               iteration < CALIBRATION_ITERATIONS; iteration++) {
            #pragma omp target teams num_teams(localTeams) \
                                     thread_limit(THREADS_PER_BLOCK)
            {
              output_t histogram[MAX_LOCAL_HISTOGRAM_BINS];

              #pragma omp parallel shared(histogram)
              {
                const int thread = omp_get_thread_num();
                const int threads = omp_get_num_threads();
                const int team = omp_get_team_num();
                const int teams = omp_get_num_teams();

                for (IndexType bin = thread; bin < nbins; bin += threads)
                  histogram[bin] = 0;

                #pragma omp barrier

                for (IndexType i = (IndexType)team * threads + thread;
                     i < input_size; i += (IndexType)teams * threads) {
                  const input_t v = input[i];
                  if (v >= minvalue && v <= maxvalue) {
                    const IndexType bin = getBin<input_t, IndexType>(
                                          v, minvalue, maxvalue, nbins);
                    #pragma omp atomic update
                    histogram[bin] += 1;
                  }
                }

                #pragma omp barrier

                for (IndexType bin = thread; bin < nbins; bin += threads) {
                  #pragma omp atomic update
                  output[bin] += histogram[bin];
                }
              }
            }
          }
        } else {
          const IndexType activePartialHistograms = activeConfiguration;
          for (int iteration = 0;
               iteration < CALIBRATION_ITERATIONS; iteration++) {
            #pragma omp target teams distribute parallel for \
                               num_teams(activePartialHistograms) \
                               thread_limit(THREADS_PER_BLOCK)
            for (IndexType linearIndex = 0;
                 linearIndex < input_size; linearIndex++) {
              const input_t v = input[linearIndex];
              if (v >= minvalue && v <= maxvalue) {
                const IndexType bin = getBin<input_t, IndexType>(
                                      v, minvalue, maxvalue, nbins);
                const IndexType team = omp_get_team_num();
                #pragma omp atomic update
                partial[(size_t)team * nbins + bin] += 1;
              }
            }

            // Threads own output bins, so the reduction needs no atomics.
            #pragma omp target teams distribute parallel for
            for (IndexType bin = 0; bin < nbins; bin++) {
              output_t sum = 0;
              for (IndexType histogram = 0;
                   histogram < activePartialHistograms; histogram++) {
                const size_t offset = (size_t)histogram * nbins + bin;
                sum += partial[offset];
                partial[offset] = 0;
              }
              output[bin] += sum;
            }
          }
        }
        const auto calibrationEnd = std::chrono::steady_clock::now();
        const long long candidateTime =
          std::chrono::duration_cast<std::chrono::nanoseconds>(
            calibrationEnd - calibrationStart).count();
        if (trial == 1 && (bestTime == 0 || candidateTime < bestTime)) {
          bestTime = candidateTime;
          bestConfiguration = activeConfiguration;
        }
      }
    }

    #pragma omp target teams distribute parallel for
    for (IndexType i = 0; i < output_size; i++) output[i] = 0;

    if (bestConfiguration == 0)
      printf("bincount using global atomics (auto-selected)\n");
    else if (bestConfiguration < 0)
      printf("bincount using team-local histogram with %d teams "
             "(auto-selected)\n", -bestConfiguration);
    else
      printf("bincount using %d global partial histograms (auto-selected)\n",
             bestConfiguration);

    const auto start = std::chrono::steady_clock::now();
    if (bestConfiguration == 0) {
      for (int iteration = 0; iteration < repeat; iteration++) {
        #pragma omp target teams distribute parallel for
        for (IndexType linearIndex = 0;
             linearIndex < input_size; linearIndex++) {
          const input_t v = input[linearIndex];
          if (v >= minvalue && v <= maxvalue) {
            const IndexType bin = getBin<input_t, IndexType>(
                                  v, minvalue, maxvalue, nbins);
            #pragma omp atomic update
            output[bin] += 1;
          }
        }
      }
    } else if (bestConfiguration < 0) {
      const IndexType localTeams = -bestConfiguration;
      for (int iteration = 0; iteration < repeat; iteration++) {
        #pragma omp target teams num_teams(localTeams) \
                                 thread_limit(THREADS_PER_BLOCK)
        {
          output_t histogram[MAX_LOCAL_HISTOGRAM_BINS];

          #pragma omp parallel shared(histogram)
          {
            const int thread = omp_get_thread_num();
            const int threads = omp_get_num_threads();
            const int team = omp_get_team_num();
            const int teams = omp_get_num_teams();

            for (IndexType bin = thread; bin < nbins; bin += threads)
              histogram[bin] = 0;

            #pragma omp barrier

            for (IndexType i = (IndexType)team * threads + thread;
                 i < input_size; i += (IndexType)teams * threads) {
              const input_t v = input[i];
              if (v >= minvalue && v <= maxvalue) {
                const IndexType bin = getBin<input_t, IndexType>(
                                      v, minvalue, maxvalue, nbins);
                #pragma omp atomic update
                histogram[bin] += 1;
              }
            }

            #pragma omp barrier

            for (IndexType bin = thread; bin < nbins; bin += threads) {
              #pragma omp atomic update
              output[bin] += histogram[bin];
            }
          }
        }
      }
    } else {
      const IndexType bestPartialHistograms = bestConfiguration;
      for (int iteration = 0; iteration < repeat; iteration++) {
        #pragma omp target teams distribute parallel for \
                           num_teams(bestPartialHistograms) \
                           thread_limit(THREADS_PER_BLOCK)
        for (IndexType linearIndex = 0;
             linearIndex < input_size; linearIndex++) {
          const input_t v = input[linearIndex];
          if (v >= minvalue && v <= maxvalue) {
            const IndexType bin = getBin<input_t, IndexType>(
                                  v, minvalue, maxvalue, nbins);
            const IndexType team = omp_get_team_num();
            #pragma omp atomic update
            partial[(size_t)team * nbins + bin] += 1;
          }
        }

        #pragma omp target teams distribute parallel for
        for (IndexType bin = 0; bin < nbins; bin++) {
          output_t sum = 0;
          for (IndexType histogram = 0;
               histogram < bestPartialHistograms; histogram++) {
            const size_t offset = (size_t)histogram * nbins + bin;
            sum += partial[offset];
            partial[offset] = 0;
          }
          output[bin] += sum;
        }
      }
    }
    const auto end = std::chrono::steady_clock::now();
    const long long time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(
        end - start).count();
    printf("Average execution time of bincount kernel: %f (us)\n",
           (time * 1e-3f) / repeat);
    }

    int status = memcmp(output, output_r, output_size_bytes);
    printf("%s\n", status ? "FAIL" : "PASS");

    free(partial);

    free(output);
    free(output_r);
  }
  }
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
