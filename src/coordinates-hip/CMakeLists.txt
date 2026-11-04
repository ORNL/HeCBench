# coordinates-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME coordinates
    MODEL hip
    SOURCES main.cu
    CATEGORIES geoscience
)
