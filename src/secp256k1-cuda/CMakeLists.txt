# secp256k1-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME secp256k1
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
