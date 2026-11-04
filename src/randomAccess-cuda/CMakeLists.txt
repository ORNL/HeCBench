# randomAccess-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME randomAccess
    MODEL cuda
    SOURCES main.cu
    CATEGORIES bandwidth
)
