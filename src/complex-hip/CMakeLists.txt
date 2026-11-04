# complex-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME complex
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
