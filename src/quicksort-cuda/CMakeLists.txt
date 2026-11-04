# quicksort-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME quicksort
    MODEL cuda
    SOURCES main.cu
    CATEGORIES sorting
)
