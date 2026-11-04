# bitonic-sort-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME bitonic-sort
    MODEL hip
    SOURCES main.cu
    CATEGORIES sorting
)
