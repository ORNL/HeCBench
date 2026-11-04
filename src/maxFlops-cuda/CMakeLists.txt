# maxFlops-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME maxFlops
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
