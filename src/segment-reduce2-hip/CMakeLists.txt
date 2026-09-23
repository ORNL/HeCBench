# segment-reduce2-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../segment-reduce2-cuda")

add_hecbench_benchmark(
    NAME segment-reduce2
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
    INCLUDE_DIRS ${INC_DIR}
)
