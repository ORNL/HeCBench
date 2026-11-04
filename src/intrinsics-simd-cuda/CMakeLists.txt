# intrinsics-simd-cuda/CMakeLists.txt

add_hecbench_benchmark(
    NAME intrinsics-simd
    MODEL cuda
    SOURCES main.cu
    CATEGORIES language
)
