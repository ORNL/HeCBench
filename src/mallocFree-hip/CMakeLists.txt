# mallocFree-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME mallocFree
    MODEL hip
    SOURCES main.cu
    CATEGORIES language
)
