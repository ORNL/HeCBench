# gsplat3d-hip/CMakeLists.txt

set(CPU_REFERENCE "${CMAKE_CURRENT_LIST_DIR}/../gsplat3d-cuda/reference.cpp")

add_hecbench_benchmark(
  NAME gsplat3d
  MODEL hip
  SOURCES main.cu forward.cu baseline.cu optimized.cu ${CPU_REFERENCE}
  CATEGORIES imaging
  TEST_ARGS 4096 128 128 10
  TEST_REGEX "Rasterize backward: PASS"
  TEST_TIMEOUT 300
)
