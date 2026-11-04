# tsa-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME tsa
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
