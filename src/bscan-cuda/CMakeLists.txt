# bscan-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME bscan
    MODEL cuda
    SOURCES main.cu
    CATEGORIES compression
)
