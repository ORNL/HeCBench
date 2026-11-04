# bitpermute-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME bitpermute
    MODEL hip
    SOURCES main.cu
    CATEGORIES cryptography
)
