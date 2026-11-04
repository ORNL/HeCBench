# crossEntropy-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME crossEntropy
    MODEL hip
    SOURCES main.cu
    CATEGORIES ml
)
