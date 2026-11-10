# pointwise-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME pointwise
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
