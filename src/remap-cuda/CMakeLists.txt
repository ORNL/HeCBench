# remap-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME remap
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
