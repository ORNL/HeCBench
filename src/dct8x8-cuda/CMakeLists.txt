# dct8x8-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME dct8x8
    MODEL cuda
    SOURCES DCT8x8_gold.cu kernels.cu main.cu
    CATEGORIES imaging
)
