# streamUM-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME streamUM
    MODEL hip
    SOURCES main.cu
    CATEGORIES language
    COMPILE_OPTIONS -fopenmp 
    LINK_LIBRARIES hipblas -fopenmp
)
