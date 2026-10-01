# asyncAllocation-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME asyncAllocation
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language_features
)
