# ecdh-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME ecdh
    MODEL cuda
    SOURCES main.cu
    CATEGORIES cryptography
)
