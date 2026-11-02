# index-shuffle-cuda/CMakeLists.txt

add_hecbench_benchmark(
  NAME index-shuffle
  MODEL cuda
  SOURCES main.cu
  CATEGORIES algorithms
  TEST_ARGS 2048 128 4 1
  TEST_REGEX "PASS"
)
