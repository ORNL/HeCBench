# simpleSpmv-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME simpleSpmv
    MODEL cuda
    SOURCES kernels.cu main.cpp utils.cpp
    CATEGORIES math
)
