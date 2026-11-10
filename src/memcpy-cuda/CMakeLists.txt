# memcpy-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME memcpy
    MODEL cuda
    SOURCES main.cu
    CATEGORIES bandwidth
)
