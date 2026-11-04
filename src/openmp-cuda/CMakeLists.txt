# openmp-cuda/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../openmp-omp")

add_hecbench_benchmark(
    NAME openmp
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
    INCLUDE_DIRS ${INC_DIR}
    COMPILE_OPTIONS -Xcompiler -fopenmp
    LINK_LIBRARIES -fopenmp 
)
