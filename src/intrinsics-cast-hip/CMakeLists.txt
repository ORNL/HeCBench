# intrinsics-cast-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME intrinsics-cast
    MODEL hip
    SOURCES main.cu
    CATEGORIES language
)
