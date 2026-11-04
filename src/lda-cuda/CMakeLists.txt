# lda-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME lda
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
