# gelu-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME gelu
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
