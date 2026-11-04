# dct8x8-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME dct8x8
    MODEL hip
    SOURCES DCT8x8_gold.cu kernels.cu main.cu
    CATEGORIES imaging
)
