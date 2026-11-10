# intrinsics-cast-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME intrinsics-cast
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
