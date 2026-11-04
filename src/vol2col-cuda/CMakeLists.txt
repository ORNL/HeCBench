# vol2col-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME vol2col
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
