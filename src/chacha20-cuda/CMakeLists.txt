# chacha20-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME chacha20
    MODEL cuda
    SOURCES main.cu
    CATEGORIES cryptography
)
