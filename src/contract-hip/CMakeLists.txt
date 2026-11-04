# contract-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME contract
    MODEL hip
    SOURCES main.cu
    CATEGORIES compression
)
