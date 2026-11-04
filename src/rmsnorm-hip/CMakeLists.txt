# rmsnorm-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME rmsnorm
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
