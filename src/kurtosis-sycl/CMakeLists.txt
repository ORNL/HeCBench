# kurtosis-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME kurtosis
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES math
)
