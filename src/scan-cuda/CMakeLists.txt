# scan-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME scan
    MODEL cuda
    SOURCES main.cu
    CATEGORIES compression
)
