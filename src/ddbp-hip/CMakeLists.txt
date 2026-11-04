# ddbp-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME ddbp
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
