# interleave-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME interleave
    MODEL hip
    SOURCES main.cu
    CATEGORIES language
)
