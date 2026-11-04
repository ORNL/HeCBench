# eigenvalue-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME eigenvalue
    MODEL cuda
    SOURCES main.cu reference.cu
    CATEGORIES math
)
