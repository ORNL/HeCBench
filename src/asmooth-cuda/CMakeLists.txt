# asmooth-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME asmooth
    MODEL cuda
    SOURCES main.cu
    CATEGORIES imaging
)
