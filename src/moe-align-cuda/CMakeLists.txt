# moe-align-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME moe-align
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
