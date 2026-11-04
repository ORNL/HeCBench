# dct8x8-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME dct8x8
    MODEL sycl
    SOURCES DCT8x8_gold.cpp kernels.cpp main.cpp
    CATEGORIES imaging
)
