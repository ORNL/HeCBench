# fpc-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME fpc
    MODEL cuda
    SOURCES main.cu
    CATEGORIES compression
)
