# jaccard-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME jaccard
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
