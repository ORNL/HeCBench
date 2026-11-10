# cross-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME cross
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
