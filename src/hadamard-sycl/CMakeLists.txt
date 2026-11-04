# hadamard-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME hadamard
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
