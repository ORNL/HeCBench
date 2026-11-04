# overlap-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME overlap
    MODEL hip
    SOURCES main.cu
    CATEGORIES language
)
