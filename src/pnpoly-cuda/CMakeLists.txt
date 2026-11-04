# pnpoly-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME pnpoly
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
