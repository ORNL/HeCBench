# chemv-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME chemv
    MODEL hip
    SOURCES kernel.cu main.cu
    CATEGORIES simulation
)
