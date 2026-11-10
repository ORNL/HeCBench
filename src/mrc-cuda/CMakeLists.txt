# mrc-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME mrc
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
