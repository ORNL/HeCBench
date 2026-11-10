# rfs-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME rfs
    MODEL cuda
    SOURCES main.cu
    CATEGORIES math
)
