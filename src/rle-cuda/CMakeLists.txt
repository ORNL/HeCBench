# rle-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME rle
    MODEL cuda
    SOURCES main.cu
    CATEGORIES compression
)
