# bitonic-sort-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME bitonic-sort
    MODEL cuda
    SOURCES main.cu
    CATEGORIES sorting
)
