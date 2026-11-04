# prefetch-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME prefetch
    MODEL hip
    SOURCES main.cu
    CATEGORIES language
)
