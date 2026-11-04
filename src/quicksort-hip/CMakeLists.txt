# quicksort-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME quicksort
    MODEL hip
    SOURCES main.cu
    CATEGORIES sorting
)
