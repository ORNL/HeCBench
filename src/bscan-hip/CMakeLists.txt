# bscan-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME bscan
    MODEL hip
    SOURCES main.cu
    CATEGORIES compression
)
