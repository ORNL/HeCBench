# adjacent-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME adjacent
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
