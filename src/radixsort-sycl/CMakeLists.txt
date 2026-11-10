# radixsort-sycl/CMakeLists.txt

add_hecbench_benchmark(
    NAME radixsort
    MODEL sycl
    SOURCES RadixSort.cpp Scan.cpp main.cpp
    CATEGORIES sorting
)
