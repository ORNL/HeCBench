# present-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME present
    MODEL hip
    SOURCES main.cu
    CATEGORIES cryptography
)
