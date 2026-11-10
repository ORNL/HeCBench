# fma-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME fma
    MODEL cuda
    SOURCES main.cu
    CATEGORIES compression
)
