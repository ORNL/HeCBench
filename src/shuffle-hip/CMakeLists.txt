# shuffle-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME shuffle
    MODEL hip
    SOURCES main.cu
    CATEGORIES language
)
