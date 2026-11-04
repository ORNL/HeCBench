# bincount-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME bincount
    MODEL cuda
    SOURCES main.cu
    CATEGORIES ml
)
