# ecdh-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME ecdh
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES cryptography
)
