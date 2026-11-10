# bscan-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME bscan
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES compression
)
