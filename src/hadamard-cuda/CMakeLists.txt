# hadamard-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME hadamard
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
