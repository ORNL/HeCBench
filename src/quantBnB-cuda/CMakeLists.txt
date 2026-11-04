# quantBnB-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME quantBnB
    MODEL cuda
    SOURCES main.cu
    CATEGORIES compression
)
