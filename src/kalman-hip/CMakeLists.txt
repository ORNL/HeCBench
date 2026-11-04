# kalman-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME kalman
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
