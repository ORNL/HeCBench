#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <chrono>
#include <random>
#include <vector>
#include <omp.h>

int main(int argc, char **argv) {
  if (argc != 4) {
    printf("Usage: %s <number of elements> <block size> <repeat>\n", argv[0]);
    return 1;
  }
  const int num_elems = atoi(argv[1]);
  const int block_size = atoi(argv[2]);
  const int repeat = atoi(argv[3]);
    
  std::vector<int> input (num_elems);
  std::vector<int> output (num_elems);

  // Generate input data.
  for (int i = 0; i < num_elems; i++) {
    input[i] = i - num_elems / 2;
  }

  std::mt19937 g;
  g.seed(19937);
  std::shuffle(input.begin(), input.end(), g);

  int *data_to_filter = input.data();
  int *filtered_data = output.data();
  int nres[1];

  #pragma omp target data map(to: data_to_filter[0:num_elems]) \
                          map(from: nres[0:1]) \
                          map(from: filtered_data[0:num_elems])
  {
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < repeat; i++) {
      nres[0] = 0;
      #pragma omp target update to (nres[0:1]) // need to reset on device 

      #pragma omp target teams num_teams((num_elems+block_size-1)/block_size) \
      thread_limit(block_size) 
      {
        int l_n;
        #pragma omp parallel 
        {
          int i = omp_get_team_num() * omp_get_num_threads() + omp_get_thread_num() ;
          if (omp_get_thread_num() == 0)
            l_n = 0;
          #pragma omp barrier
          int d, pos;
        
          if(i < num_elems) {
            d = data_to_filter[i];
            if(d > 0) {
              #pragma omp atomic capture
              pos = l_n++;
            }
          }
          #pragma omp barrier
  
          // leader increments the global counter
          if (omp_get_thread_num() == 0) {
            //l_n = atomicAdd(nres, l_n);
             int old;
             #pragma omp atomic capture
             {
                old = nres[0];
                nres[0] += l_n; 
             }
             l_n = old;
          }
          #pragma omp barrier
        
          // threads with true predicates write their elements
          if(i < num_elems && d > 0) {
            pos += l_n; // increment local pos by global counter
            filtered_data[pos] = d;
          }
          #pragma omp barrier
        }
      }
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time %lf (ms)\n", (time * 1e-6) / repeat);
  }

  std::vector<int> h_output (num_elems);

  // Generate host output with host filtering code.
  int h_flt_count = 0;
  for (int i = 0; i < num_elems; i++) {
    if (input[i] > 0) {
      h_output[h_flt_count++] = input[i];
    }
  }

  // Verify
  std::sort(h_output.begin(), h_output.begin() + h_flt_count);
  std::sort(output.begin(), output.begin() + nres[0]);

  bool equal = (h_flt_count == nres[0]) && 
               std::equal(h_output.begin(),
                          h_output.begin() + h_flt_count, output.begin());

  printf("\nFilter using shared memory %s \n",
         equal ? "PASS" : "FAIL");
  if (!equal) exit(1);

  return 0;
}
