# dense-embedding-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME dense-embedding
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
