# lfib4-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME lfib4
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
