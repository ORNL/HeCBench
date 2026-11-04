# mtf-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME mtf
    MODEL hip
    SOURCES ${CMAKE_SOURCE_DIR}/src/mtf-cuda/main.cu
    CATEGORIES compression
)
