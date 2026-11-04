# dispatch-hip/CMakeLists.txt

add_hecbench_benchmark(
    NAME dispatch
    MODEL hip
    SOURCES main.cu
    CATEGORIES language
)
