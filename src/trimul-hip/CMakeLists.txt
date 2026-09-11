# trimul-hip/CMakeLists.txt

# -fopenmp parallelizes the CPU reference in reference.h
add_hecbench_benchmark(
    NAME trimul
    MODEL hip
    SOURCES main.cu
    COMPILE_OPTIONS -fopenmp
    LINK_LIBRARIES hipblas -fopenmp
    CATEGORIES ml
)
