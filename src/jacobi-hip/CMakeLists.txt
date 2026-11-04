# jacobi-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME jacobi
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
