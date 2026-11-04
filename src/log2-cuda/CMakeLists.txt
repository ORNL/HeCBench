# log2-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME log2
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
