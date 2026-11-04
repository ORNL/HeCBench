# clock-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME clock
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
