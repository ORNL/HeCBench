# radixsort-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME radixsort
    MODEL cuda
    SOURCES RadixSort.cu Scan.cu main.cu
    CATEGORIES sorting
)
