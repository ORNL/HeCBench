# dxtc2-cuda/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../boxfilter-sycl")

add_hecbench_benchmark(
    NAME dxtc2
    MODEL cuda
    SOURCES main.cu ${INC_DIR}/shrUtils.cpp
    CATEGORIES compression
    INCLUDE_DIRS ${INC_DIR}
)
