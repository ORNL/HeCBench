# glu-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME glu
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
