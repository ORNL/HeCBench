# hausdorff-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME hausdorff
    MODEL hip
    SOURCES main.cu
    CATEGORIES geoscience
)
