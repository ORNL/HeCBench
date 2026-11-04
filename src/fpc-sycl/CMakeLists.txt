# fpc-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME fpc
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES compression
)
