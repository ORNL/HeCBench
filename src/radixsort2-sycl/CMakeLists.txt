# radixsort2-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME radixsort2
    MODEL sycl
    SOURCES main.cpp
    CATEGORIES sorting
)
