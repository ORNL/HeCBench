# awbarrier-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME awbarrier
    MODEL cuda
    SOURCES main.cu
    CATEGORIES algorithms
)
