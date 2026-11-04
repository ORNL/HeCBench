# pool-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME pool
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
