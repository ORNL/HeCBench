# mtf-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME mtf
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES compression
)
