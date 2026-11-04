# lanczos-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME lanczos
    MODEL hip
    SOURCES lanczos.cu main.cu
    CATEGORIES math
)
