# wlcpow-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME wlcpow
    MODEL hip
    SOURCES main.cu
    CATEGORIES simulation
)
