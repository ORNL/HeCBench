# pool-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME pool
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
