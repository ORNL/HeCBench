# streamOrderedAllocation-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME streamOrderedAllocation
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
