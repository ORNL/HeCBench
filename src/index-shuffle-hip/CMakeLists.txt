# index-shuffle-hip/CMakeLists.txt

add_hecbench_benchmark(
  NAME index-shuffle
  MODEL hip
  SOURCES main.cu
  CATEGORIES algorithms
  TEST_ARGS 2048 128 4 1
  TEST_REGEX "PASS"
)
