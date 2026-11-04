# dropout-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME dropout
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
