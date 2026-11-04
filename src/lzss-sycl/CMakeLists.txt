# lzss-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME lzss
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES compression
)
