# relu-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME relu
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
