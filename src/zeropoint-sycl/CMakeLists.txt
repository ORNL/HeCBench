# zeropoint-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME zeropoint
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
