# entropy-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME entropy
    MODEL hip
    SOURCES main.cu
    CATEGORIES encoding
)
