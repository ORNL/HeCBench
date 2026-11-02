# index-shuffle-omp/CMakeLists.txt

add_hecbench_benchmark(
  NAME index-shuffle
  MODEL omp
  SOURCES main.cpp
  CATEGORIES algorithms
  TEST_ARGS 2048 128 4 1
  TEST_REGEX "PASS"
)
