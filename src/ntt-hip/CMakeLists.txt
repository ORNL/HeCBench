# ntt-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME ntt
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
