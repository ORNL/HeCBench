# radixsort-omp/CMakeLists.txt

add_hecbench_benchmark(
    NAME radixsort
    MODEL omp
    SOURCES main.cpp RadixSort.cpp Scan.cpp
    CATEGORIES sorting
)
