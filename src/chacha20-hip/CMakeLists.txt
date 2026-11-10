# chacha20-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME chacha20
    MODEL hip
    SOURCES main.cu
    CATEGORIES cryptography
)
