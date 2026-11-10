# gamma-correction-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME gamma-correction
    MODEL hip
    SOURCES main.cu
    CATEGORIES imaging
)
