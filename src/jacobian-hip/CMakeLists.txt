# jacobian-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME jacobian
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
