# entropy-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME entropy
    MODEL cuda
    SOURCES main.cu
    CATEGORIES encoding
)
