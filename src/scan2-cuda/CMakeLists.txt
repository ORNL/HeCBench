# scan2-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME scan2
    MODEL cuda
    SOURCES main.cu
    CATEGORIES compression
)
