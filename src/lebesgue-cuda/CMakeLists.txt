# lebesgue-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME lebesgue
    MODEL cuda
    SOURCES kernels.cu main.cpp utils.cpp
    CATEGORIES math
)
