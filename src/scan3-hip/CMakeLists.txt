# scan3-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../scan2-cuda")

add_hecbench_benchmark(
    NAME scan3
    MODEL hip
    SOURCES main.cu
    CATEGORIES compression
    INCLUDE_DIRS ${INC_DIR}
)
