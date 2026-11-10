# mr-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME mr
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
