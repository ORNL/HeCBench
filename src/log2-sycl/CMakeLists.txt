# log2-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME log2
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
