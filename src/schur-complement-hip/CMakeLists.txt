# schur-complement-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../schur-complement-cuda")

add_hecbench_benchmark(
    NAME schur-complement
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
    INCLUDE_DIRS ${INC_DIR}
)
