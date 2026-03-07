#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <omp.h>
#include "complex.h"
#include "kernels.h"

bool check (const char *cs, int n)
{
  bool ok = true;
  for (int i = 0; i < n; i++) {
    if (cs[i] != 5) {
      ok = false; 
      break;
    }
  }
  return ok;
}

int main(int argc, char* argv[]) {
  if (argc != 3) {
    printf("Usage: %s <size> <repeat>\n", argv[0]);
    return 1;
  }
  const int n = atoi(argv[1]);
  const int repeat = atoi(argv[2]);

  char* cs = (char*) malloc (n);

  #pragma omp target data map (alloc: cs[0:n])
  {
    // warmup
    complex_float(cs, n);
    complex_double(cs, n);

    auto start = std::chrono::steady_clock::now();

    // complex numbers in single precision
    for (int i = 0; i < repeat; i++) {
      complex_float(cs, n);
    }

    auto end = std::chrono::steady_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (float) %f (s)\n", time * 1e-9f / repeat);

    #pragma omp target update from (cs[0:n])

    bool complex_float_check = check(cs, n);

    start = std::chrono::steady_clock::now();

    // complex numbers in double precision
    for (int i = 0; i < repeat; i++) {
      complex_double(cs, n);
    }

    end = std::chrono::steady_clock::now();
    time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    printf("Average kernel execution time (double) %f (s)\n", time * 1e-9f / repeat);

    #pragma omp target update from (cs[0:n])
    bool complex_double_check = check(cs, n);

    printf("%s\n", (complex_float_check && complex_double_check)
                   ? "PASS" : "FAIL");
  if (!(complex_float_check && complex_double_check)) exit(1);
  }

  free(cs);

  return 0;
}
