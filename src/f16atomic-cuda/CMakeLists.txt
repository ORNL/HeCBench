# f16atomic-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME f16atomic
    MODEL cuda
    SOURCES main.cu
    CATEGORIES compression
)
