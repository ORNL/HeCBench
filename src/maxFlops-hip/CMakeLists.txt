# maxFlops-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME maxFlops
    MODEL hip
    SOURCES main.cu
    CATEGORIES language
)
