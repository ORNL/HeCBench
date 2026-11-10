# tsa-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME tsa
    MODEL cuda
    SOURCES main.cu
    CATEGORIES simulation
)
