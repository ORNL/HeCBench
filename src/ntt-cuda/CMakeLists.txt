# ntt-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME ntt
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
