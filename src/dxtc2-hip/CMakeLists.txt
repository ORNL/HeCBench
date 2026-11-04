# dxtc2-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../boxfilter-sycl")

add_hecbench_benchmark(
    NAME dxtc2
    MODEL hip
    SOURCES main.cu ${INC_DIR}/shrUtils.cpp
    CATEGORIES compression
    INCLUDE_DIRS ${INC_DIR}
)
