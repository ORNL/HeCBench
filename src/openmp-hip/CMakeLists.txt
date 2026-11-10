# openmp-hip/CMakeLists.txt

set(INC_DIR "${CMAKE_CURRENT_LIST_DIR}/../openmp-omp")

add_hecbench_benchmark(
    NAME openmp
    MODEL hip
    SOURCES main.cu
    INCLUDE_DIRS ${INC_DIR}
    CATEGORIES language
    COMPILE_OPTIONS -fopenmp 
    LINK_LIBRARIES hipblas -fopenmp
)
