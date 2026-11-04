# recursiveGaussian-omp/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../boxfilter-sycl")
set(INC2_DIR "${CMAKE_CURRENT_LIST_DIR}/../recursiveGaussian-cuda")

add_hecbench_benchmark(
    NAME recursiveGaussian
    MODEL omp
    SOURCES main.cpp
            ${INC_DIR}/shrUtils.cpp
            ${INC2_DIR}/RecursiveGaussianHost.cpp
    CATEGORIES imaging
    INCLUDE_DIRS ${INC_DIR} ${INC2_DIR}
)
