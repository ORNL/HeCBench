# mixbench-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME mixbench
    MODEL hip
    SOURCES main.cu
    CATEGORIES language
)
