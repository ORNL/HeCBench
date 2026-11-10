# svd3x3-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME svd3x3
    MODEL hip
    SOURCES main.cu
    COMPILE_OPTIONS -ffast-math
    CATEGORIES math
)
