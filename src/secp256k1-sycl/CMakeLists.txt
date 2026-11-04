# secp256k1-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME secp256k1
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
