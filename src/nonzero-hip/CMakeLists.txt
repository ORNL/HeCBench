# nonzero-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME nonzero
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
