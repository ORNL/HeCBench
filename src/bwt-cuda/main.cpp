#include <string>
#include <vector>
#include <list>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include "bwt.hpp"

#define NOW std::chrono::high_resolution_clock::now()

std::string bwt_cpu(const std::string sequence) {
  const int n = sequence.size();
  const char* sequence_cstr = sequence.c_str();

  std::vector<int> table(n);

  for (int i = 0; i < n; i++){
    table[i] = i;
  }

  std::list<int> sorted_table(table.begin(), table.end());
  sorted_table.sort([sequence_cstr,n](const int& a, const int& b) -> bool {
    for(int i = 0; i < n; i++) {
      if(sequence_cstr[(a + i) % n] != sequence_cstr[(b + i) % n]) {
        return sequence_cstr[(a + i) % n] < sequence_cstr[(b + i) % n];
      }
    }
    return false;
  });

  std::string transformed_sequence;

  for(auto r = sorted_table.begin(); r != sorted_table.end(); ++r) {
    transformed_sequence += sequence_cstr[(n + *r - 1) % n];
  }
  return transformed_sequence;
}

int main(int argc, char const *argv[])
{
  const int N = (argc > 1) ? atoi(argv[1]) : 1E6;
  std::cout << "running a sample sequence of length " << N << std::endl;

  // initialize the alphabet used in bioinformatics
  std::string alphabet("ATCG");

  srand(123);
  char* sequence = (char*) malloc((N+1) * sizeof(char));

  for (int i = 0; i < N; i++) {
    sequence[i] = alphabet[rand() % alphabet.size()];
  }
  sequence[N] = ETX;

  // host run may take a while
  auto start = NOW;
  auto cpu_seq = bwt_cpu(sequence);
  auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(NOW - start);

  // device run
  start = NOW;
  auto gpu_seq = bwt(sequence);
  auto gpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(NOW - start);

  std::cout << "Host time: " << cpu_time.count() << " ms" << std::endl;
  std::cout << "Device time: " << gpu_time.count() << " ms" << std::endl;

  if(cpu_seq.compare(gpu_seq) == 0) {
    std::cout << "PASS\n";
  } else {
    std::cout << "FAIL\n";
    exit(1);
  }

  free(sequence);
  return 0;
}
