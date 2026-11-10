# scan-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME scan
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES compression
)
