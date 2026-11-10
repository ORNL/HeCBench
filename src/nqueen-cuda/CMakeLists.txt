# nqueen-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME nqueen
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
