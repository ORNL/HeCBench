# mtf-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME mtf
    MODEL cuda
    SOURCES main.cu
    CATEGORIES compression
)
