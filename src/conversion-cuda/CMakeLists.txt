# conversion-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME conversion
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
