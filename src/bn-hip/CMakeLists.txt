# bn-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME bn
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
