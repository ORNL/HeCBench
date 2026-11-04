# medianfilter-cuda/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../boxfilter-sycl")

add_hecbench_benchmark(
    NAME medianfilter
    MODEL cuda
    SOURCES MedianFilterHost.cpp main.cu ${INC_DIR}/shrUtils.cpp
    CATEGORIES imaging
    INCLUDE_DIRS ${INC_DIR}
)
