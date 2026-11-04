# blockAccess-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME blockAccess
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
