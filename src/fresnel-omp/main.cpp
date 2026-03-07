#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <omp.h>

#pragma omp declare target
double Fresnel_Sine_Integral(double);
#pragma omp end declare target

void reference (const double *__restrict input, double *__restrict output, const int n) {
  for (int i = 0; i < n; i++)
    output[i] = Fresnel_Sine_Integral(input[i]);
}

int main(int argc, char *argv[])
{
  if (argc != 2) {
    printf("Usage: %s <repeat>\n", argv[0]);
    return 1;
  }
  const int repeat = atoi(argv[1]);

  // range [0, 8], interval 1e-7
  const double interval = 1e-7;
  const int points = (int)(8.0 / interval);
  const size_t points_size = points * sizeof(double);
  double *x = (double*) malloc (points_size);
  double *output = (double*) malloc (points_size);
  double *h_output = (double*) malloc (points_size);
  for (int i = 0; i < points; i++)
    x[i] = (double)i * interval;
	   
#pragma omp target data map(x[0:points]) map(from: output[0:points])
{
  auto start = std::chrono::steady_clock::now();

  for (int i = 0; i < repeat; i++) {
    #pragma omp target teams distribute parallel for thread_limit(256)
    for (int i = 0; i < points; i++) 
      output[i] = Fresnel_Sine_Integral(x[i]);
  }

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Average kernel execution time %f (s)\n", (time * 1e-9f) / repeat);
}

  // verify
  reference(x, h_output, points);
  bool ok = true;
  for (int i = 0; i < points; i++) {
    if (fabs(h_output[i] - output[i]) > 1e-6) {
      printf("%lf %lf\n", h_output[i], output[i]);
      ok = false;
      break;
    }
  }
  printf("%s\n", ok ? "PASS" : "FAIL");
  
  free(x);
  free(output);
  free(h_output);
  if (!ok) return 1;
  return 0;
}
