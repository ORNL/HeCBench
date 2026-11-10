# atomicReduction-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME atomicReduction
    MODEL hip
    SOURCES reduction.cu
    CATEGORIES compression
)
