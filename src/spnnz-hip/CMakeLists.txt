# spnnz-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../spd2s-cuda")

add_hecbench_benchmark(
    NAME spnnz
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
    INCLUDE_DIRS ${INC_DIR}
    LINK_LIBRARIES hipsparse
)
