# rmsnorm-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME rmsnorm
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
