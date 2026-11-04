# kiss-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME kiss
    MODEL cuda
    SOURCES main.cu
    CATEGORIES encoding
)
