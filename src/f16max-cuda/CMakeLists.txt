# f16max-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME f16max
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
