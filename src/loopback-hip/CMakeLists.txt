# loopback-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME loopback
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
