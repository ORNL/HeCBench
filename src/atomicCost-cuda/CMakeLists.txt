# atomicCost-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME atomicCost
    MODEL cuda
    SOURCES main.cu
    CATEGORIES compression
)
