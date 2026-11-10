# warpexchange-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../blockAccess-cuda")

add_hecbench_benchmark(
    NAME warpexchange
    MODEL hip
    SOURCES main.cu
    INCLUDE_DIRS ${INC_DIR}
    CATEGORIES language
)
