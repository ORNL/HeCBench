# mr-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME mr
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
