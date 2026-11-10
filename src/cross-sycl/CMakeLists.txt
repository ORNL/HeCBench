# cross-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME cross
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
