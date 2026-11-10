# bitpacking-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME bitpacking
    MODEL sycl
    SOURCES kernels.cpp main.cpp utils.cpp
    CATEGORIES compression
)
