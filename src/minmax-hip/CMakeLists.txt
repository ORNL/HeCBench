# minmax-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME minmax
    MODEL hip
    SOURCES ${CMAKE_SOURCE_DIR}/src/minmax-cuda/main.cu
    CATEGORIES compression
)
