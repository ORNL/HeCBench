# vadd-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME vadd
    MODEL cuda
    SOURCES main.cu
    CATEGORIES bandwidth
)
