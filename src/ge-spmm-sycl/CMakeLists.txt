# ge-spmm-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME ge-spmm
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES ml
)
