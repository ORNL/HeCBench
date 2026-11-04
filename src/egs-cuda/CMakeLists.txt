# egs-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME egs
    MODEL cuda
    SOURCES init.cu kernels.cu main.cu
    CATEGORIES imaging
)
