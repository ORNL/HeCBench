# heat2d-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME heat2d
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
