# asyncAllocation-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME asyncAllocation
    MODEL hip
    SOURCES main.cu
    CATEGORIES language_features
)
