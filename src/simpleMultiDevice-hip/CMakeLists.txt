# simpleMultiDevice-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME simpleMultiDevice
    MODEL hip
    SOURCES main.cu
    CATEGORIES language
)
