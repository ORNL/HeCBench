# streamOrderedAllocation-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME streamOrderedAllocation
    MODEL hip
    SOURCES main.cu
    CATEGORIES language
)
