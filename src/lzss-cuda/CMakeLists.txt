# lzss-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME lzss
    MODEL cuda
    SOURCES main.cu
    CATEGORIES compression
)
