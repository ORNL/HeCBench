# gsplat3d-cuda/CMakeLists.txt

add_hecbench_benchmark(
  NAME gsplat3d
  MODEL cuda
  SOURCES main.cu forward.cu baseline.cu optimized.cu reference.cpp
  CATEGORIES imaging
  TEST_ARGS 4096 128 128 10
  TEST_REGEX "Rasterize backward: PASS"
  TEST_TIMEOUT 300
)
