# matern-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME matern
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
