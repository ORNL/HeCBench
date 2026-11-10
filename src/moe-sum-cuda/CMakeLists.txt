# moe-sum-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME moe-sum
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
