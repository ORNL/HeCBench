# chacha20-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME chacha20
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES cryptography
)
