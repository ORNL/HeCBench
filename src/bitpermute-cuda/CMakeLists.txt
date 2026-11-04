# bitpermute-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME bitpermute
    MODEL cuda
    SOURCES main.cu
    CATEGORIES cryptography
)
