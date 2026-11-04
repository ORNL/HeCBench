# openmp-sycl/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../openmp-omp")

add_hecbench_benchmark(
    NAME openmp
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
    INCLUDE_DIRS ${INC_DIR}
    COMPILE_OPTIONS -fopenmp
    LINK_LIBRARIES -fopenmp 
)
