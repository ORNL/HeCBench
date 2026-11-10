# dct8x8-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME dct8x8
    MODEL omp
    SOURCES DCT8x8_gold.cpp kernels.cpp main.cpp
    CATEGORIES imaging
)
