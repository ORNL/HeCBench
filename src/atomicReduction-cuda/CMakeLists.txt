# atomicReduction-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME atomicReduction
    MODEL cuda
    SOURCES reduction.cu
    CATEGORIES compression
)
