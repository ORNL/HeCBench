# tensorT-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME tensorT
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
