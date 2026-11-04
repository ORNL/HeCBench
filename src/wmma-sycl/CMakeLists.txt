# wmma-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME wmma
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES language
)
