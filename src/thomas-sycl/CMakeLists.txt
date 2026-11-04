# thomas-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME thomas
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
