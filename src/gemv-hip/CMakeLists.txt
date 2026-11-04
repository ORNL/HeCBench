# gemv-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME gemv
    MODEL hip
    SOURCES gemv.cu main.cu
    CATEGORIES math
)
