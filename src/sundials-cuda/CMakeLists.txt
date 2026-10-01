# sundials-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME sundials
    MODEL cuda
    SOURCES kernels.cu main.cpp utils.cpp
    CATEGORIES simulation algorithms math
)
