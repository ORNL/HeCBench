# thomas-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME thomas
    MODEL hip
    SOURCES cuThomasBatch.cu main.cu
    CATEGORIES math
)
