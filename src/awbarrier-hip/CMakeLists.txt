# awbarrier-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME awbarrier
    MODEL hip
    SOURCES main.cu
    CATEGORIES algorithms
)
