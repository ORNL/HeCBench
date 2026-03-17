# vadd-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME vadd
    MODEL hip
    SOURCES main.cu
    CATEGORIES bandwidth
)
