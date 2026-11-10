# radixsort2-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME radixsort2
    MODEL cuda
    SOURCES main.cu
    CATEGORIES sorting
)
