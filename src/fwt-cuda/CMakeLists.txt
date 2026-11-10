# fwt-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME fwt
    MODEL cuda
    SOURCES main.cu reference.cu
    CATEGORIES math
)
