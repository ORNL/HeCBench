# scan2-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME scan2
    MODEL hip
    SOURCES main.cu
    CATEGORIES compression
)
