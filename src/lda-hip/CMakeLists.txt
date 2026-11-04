# lda-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME lda
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
