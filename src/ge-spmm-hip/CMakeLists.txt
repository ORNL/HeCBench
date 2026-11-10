# ge-spmm-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME ge-spmm
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
