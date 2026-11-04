# bilateral-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME bilateral
    MODEL cuda
    SOURCES main.cu
    CATEGORIES imaging
)
