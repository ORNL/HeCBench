# matrix-rotate-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME matrix-rotate
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
