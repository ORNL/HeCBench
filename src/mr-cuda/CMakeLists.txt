# mr-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME mr
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
