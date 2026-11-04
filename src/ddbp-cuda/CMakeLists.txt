# ddbp-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME ddbp
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
