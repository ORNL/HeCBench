# divergence-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME divergence
    MODEL hip
    SOURCES divergence.cu
    CATEGORIES math
)
