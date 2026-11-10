# wyllie-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME wyllie
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
