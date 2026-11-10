# reverse2D-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME reverse2D
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
