# nbnxm-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME nbnxm
    MODEL cuda
    SOURCES main.cu
    CATEGORIES bioinformatics
)
