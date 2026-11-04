# sort-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME sort
    MODEL cuda
    SOURCES main.cu
    CATEGORIES sorting
)
