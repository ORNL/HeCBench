# segment-reduce-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME segment-reduce
    MODEL hip
    SOURCES main.cu
    CATEGORIES compression
)
