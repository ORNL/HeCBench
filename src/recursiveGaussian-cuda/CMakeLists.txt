# recursiveGaussian-cuda/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../boxfilter-sycl")

add_hecbench_benchmark(
    NAME recursiveGaussian
    MODEL cuda
    SOURCES RecursiveGaussianHost.cpp main.cu ${INC_DIR}/shrUtils.cpp
    INCLUDE_DIRS ${INC_DIR}
    CATEGORIES imaging
)
