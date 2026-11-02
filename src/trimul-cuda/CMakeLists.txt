# trimul-cuda/CMakeLists.txt

# -fopenmp parallelizes the CPU reference in reference.h
add_hecbench_benchmark(
    NAME trimul
    MODEL cuda
    SOURCES main.cu
    COMPILE_OPTIONS -Xcompiler -fopenmp
    LINK_LIBRARIES cublas -fopenmp
    CATEGORIES ml
)
