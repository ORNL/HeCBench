# bincount-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME bincount
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
