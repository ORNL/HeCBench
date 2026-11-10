# sampling-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME sampling
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
