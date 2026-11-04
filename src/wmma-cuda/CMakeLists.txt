# wmma-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME wmma
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
