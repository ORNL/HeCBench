# lrn-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME lrn
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
