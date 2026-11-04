# reverse-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME reverse
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
