# phmm-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME phmm
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
