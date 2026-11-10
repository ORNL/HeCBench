# interval-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME interval
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
