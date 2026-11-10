# scan2-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME scan2
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES compression
)
