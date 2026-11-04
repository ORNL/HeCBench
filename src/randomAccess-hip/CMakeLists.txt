# randomAccess-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME randomAccess
    MODEL hip
    SOURCES main.cu
    CATEGORIES bandwidth
)
