# ge-spmm-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME ge-spmm
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
