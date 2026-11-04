# laplace-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME laplace
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
