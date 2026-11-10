# recursiveGaussian-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../boxfilter-sycl")
set(INC_DIR2 "${CMAKE_CURRENT_LIST_DIR}/../recursiveGaussian-cuda")

add_hecbench_benchmark(
    NAME recursiveGaussian
    MODEL hip
    SOURCES ${INC_DIR2}/RecursiveGaussianHost.cpp main.cu ${INC_DIR}/shrUtils.cpp
    INCLUDE_DIRS ${INC_DIR}
    CATEGORIES imaging
)
