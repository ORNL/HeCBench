# geglu-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME geglu
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
