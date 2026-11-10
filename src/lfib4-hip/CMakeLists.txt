# lfib4-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME lfib4
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
