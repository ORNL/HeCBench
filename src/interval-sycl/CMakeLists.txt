# interval-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME interval
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
