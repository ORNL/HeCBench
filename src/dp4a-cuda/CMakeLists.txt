# dp4a-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME dp4a
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
