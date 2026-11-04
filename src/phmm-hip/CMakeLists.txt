# phmm-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME phmm
    MODEL hip
    SOURCES main.cu
    CATEGORIES math
)
