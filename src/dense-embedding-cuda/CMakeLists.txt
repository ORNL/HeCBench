# dense-embedding-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME dense-embedding
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
