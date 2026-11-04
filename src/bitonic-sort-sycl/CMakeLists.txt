# bitonic-sort-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME bitonic-sort
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES sorting
)
