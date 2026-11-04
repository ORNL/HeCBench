# flame-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME flame
    MODEL hip
    SOURCES main.cu utils.cu
    CATEGORIES imaging
)
