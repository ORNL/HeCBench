# putget-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME putget
    MODEL hip
    SOURCES main.cu
    CATEGORIES bandwidth
)
