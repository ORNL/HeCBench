# bilateral-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME bilateral
    MODEL hip
    SOURCES main.cu
    CATEGORIES imaging
)
