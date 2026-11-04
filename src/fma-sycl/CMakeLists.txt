# fma-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME fma
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES compression
)
