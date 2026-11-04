# blockScan-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME blockScan
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
