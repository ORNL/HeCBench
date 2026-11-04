# lif-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME lif
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
