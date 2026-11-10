# minmax-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME minmax
    MODEL cuda
    SOURCES main.cu
    CATEGORIES compression
)
