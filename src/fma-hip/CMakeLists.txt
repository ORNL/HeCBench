# fma-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME fma
    MODEL hip
    SOURCES main.cu
    CATEGORIES compression
)
