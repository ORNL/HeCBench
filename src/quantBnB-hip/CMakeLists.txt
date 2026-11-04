# quantBnB-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME quantBnB
    MODEL hip
    SOURCES main.cu
    CATEGORIES compression
)
