# bitpermute-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME bitpermute
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES cryptography
)
