# dropout-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME dropout
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
