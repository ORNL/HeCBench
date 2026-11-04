# radixsort2-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME radixsort2
    MODEL hip
    SOURCES main.cu
    CATEGORIES sorting
)
