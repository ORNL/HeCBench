# michalewicz-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME michalewicz
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
