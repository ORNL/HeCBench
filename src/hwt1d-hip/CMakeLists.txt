# hwt1d-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME hwt1d
    MODEL hip
    SOURCES kernel.cu main.cu reference.cu
    CATEGORIES simulation
)
