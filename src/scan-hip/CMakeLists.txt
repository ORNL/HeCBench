# scan-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME scan
    MODEL hip
    SOURCES main.cu
    CATEGORIES compression
)
