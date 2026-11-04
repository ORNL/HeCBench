# winograd-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME winograd
    MODEL cuda
    SOURCES main.cu utils.cu
    CATEGORIES ml
)
