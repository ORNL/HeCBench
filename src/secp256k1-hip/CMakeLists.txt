# secp256k1-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME secp256k1
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
