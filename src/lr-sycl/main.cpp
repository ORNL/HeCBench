#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <cmath>
#include "linear.h"

extern int cpu_offset;

/* Read file */
static void create_dataset(linear_param_t * params, data_t * dataset) {
  FILE *ptr_file = fopen(params->filename, "r");
  if (ptr_file == NULL) {
    perror("Failed to load dataset file");
    exit(1);
  }

  char *token;
  char buf[1024];

  for (size_t i = 0; i < params->size && fgets(buf, 1024, ptr_file) != NULL; i++) {
    token = strtok(buf, "\t");
    dataset[i].x() = atof(token);
    token = strtok(NULL, "\t");
    dataset[i].y() = atof(token);
  }

  fclose(ptr_file);
}

static void temperature_regression(results_t * results, int repeat) {
  linear_param_t params;
  params.repeat = repeat;
  params.filename = TEMP_FILENAME;
  params.size = TEMP_SIZE;
  params.wg_size = TEMP_WORKGROUP_SIZE;
  params.wg_count = TEMP_WORKGROUP_NBR;

  data_t *dataset = (data_t*) malloc (sizeof(data_t) * params.size);
  create_dataset(&params, dataset);

  results->parallelized.ktime = 0;

  parallelized_regression(&params, dataset, &results->parallelized);
  iterative_regression(&params, dataset, &results->iterative);

  free(dataset);
}

static void print_results(results_t * results) {
  PRINT_RESULT("Parallelized", results->parallelized);
  PRINT_RESULT("Iterative", results->iterative);
}

static void write_results(results_t * results, const char * restricts) {
  FILE* file = fopen(RESULT_FILENAME, restricts);
  WRITE_RESULT(file, results->parallelized);
  WRITE_RESULT(file, results->iterative);
  fclose(file);
}

static void compare_results(const results_t * results) {
  const float tol = 1e-3f;
  bool ok = std::fabs(results->iterative.a0 - results->parallelized.a0) <=
              tol * std::fabs(results->iterative.a0) &&
            std::fabs(results->iterative.a1 - results->parallelized.a1) <=
              tol * std::fabs(results->iterative.a1);
  printf("%s\n", ok ? "PASS" : "FAIL");
}

int main(int argc, char* argv[]) {
  results_t results = {{0}};
  if (argc != 3) {
    printf("Usage: %s <repeat> <cpu offset>\n", argv[0]);
    printf("Device execution only when cpu offset is 0\n");
    printf("Host execution only when cpu offset is 100\n");
    exit(0);
  }

  int repeat = atoi(argv[1]);
  cpu_offset = atoi(argv[2]);
  printf("CPU offset: %d\n", cpu_offset);

  auto start = std::chrono::steady_clock::now();

  temperature_regression(&results, repeat);

  auto end = std::chrono::steady_clock::now();
  auto time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
  printf("Total execution time: %lf ms\n", time * 1e-6);

  printf("Average kernel execution time: %lf us\n",
         results.parallelized.ktime * 1e-3 / repeat);

  write_results(&results, "a");

  printf("\n> TEMPERATURE REGRESSION (%d)\n\n", TEMP_SIZE);
  print_results(&results);

  compare_results(&results);

  return 0;
}
