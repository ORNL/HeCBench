# radixsort-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME radixsort
    MODEL hip
    SOURCES RadixSort.cu Scan.cu main.cu
    CATEGORIES sorting
)
