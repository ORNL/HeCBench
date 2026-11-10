# ecdh-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME ecdh
    MODEL hip
    SOURCES main.cu
    CATEGORIES cryptography
)
