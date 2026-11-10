# hexciton-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME hexciton
    MODEL hip
    SOURCES main.cu reference.cu utils.cu
    CATEGORIES simulation
)
