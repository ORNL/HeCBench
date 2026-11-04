# fpc-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME fpc
    MODEL hip
    SOURCES main.cu
    CATEGORIES compression
)
